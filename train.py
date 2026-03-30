#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import transformers
# 将 Hugging Face 的日志级别设置为 ERROR，屏蔽掉普通的 WARNING 和 INFO
transformers.logging.set_verbosity_error()

#Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.utils import init_distributed_mode, is_main_process
from datetime import datetime

import wandb

best_auc = 0
best_hm = 0

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    init_distributed_mode(args)    
    device = torch.device(args.device)

    load_args(args.config, args)

    patience = args.patience
    epochs_since_improvement = 0
    best_val_hm = 0.0
    logpath = os.path.join(args.cv_dir, args.name)
    if is_main_process():
        # 1. 动态生成实验名称 (例如: zappos_layer0_mit_large)
        safe_name = args.name.replace('/', '_')
        current_time = datetime.now().strftime("%m%d_%H%M") # 精简了一下时间格式，比如 0330_1107
        # 💡 新增：获取 SLURM 集群的任务编号。如果是在本地或者普通终端直接跑的，就默认叫 'local'
        job_id = os.environ.get('SLURM_JOB_ID', 'local')       
        # 将 Job ID 拼接到最终的名称最前面
        custom_run_name = f"Job{job_id}_{args.dataset}_{current_time}_{safe_name}"

        # 如果分布式已初始化，获取总卡数；否则说明是单卡运行，卡数为 1
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        total_bs = args.batch_size * num_gpus

        # 2. 💡 新增：动态生成标签列表
        # 您可以把所有需要分类的关键属性都作为标签加进来
        run_tags = [
            args.dataset,                  # 数据集标签 (如 zappos)
            f"单卡BS_{args.batch_size}",   # 单卡 Batch Size
            f"卡数_{num_gpus}",            # 自动记录使用了几张显卡
            f"总BS_{total_bs}",            # 自动计算并记录真实生效的总 Batch Size
            "三分支基线"                   # 实验性质标签
        ]

        # 2. 初始化 wandb，使用自定义名称，并记录全部参数
        run = wandb.init(
            project="CAILA",
            name=custom_run_name,
            tags=run_tags, 
            config=vars(args)
        )
        run_name = run.name # 获取 wandb 最终确定的名字
        
        # 3. 将 run_name 作为子文件夹，生成专属保存路径
        logpath = os.path.join(args.cv_dir, safe_name, run_name)
        os.makedirs(logpath, exist_ok=True)
        
        save_args(args, logpath, args.config)
        writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
        print(f"\n[*] 实验 [{run_name}] 已启动！所有的 Checkpoints 和 logs 都将保存在: {logpath}\n")
    else:
        writer = None
        logpath = None

   
    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        pair_dropout=args.pair_dropout,
        train_only= args.train_only,
        open_world=args.open_world,
        norm_family=args.norm_family,
        dataset=args.dataset
    )
    if args.distributed:
        sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=False)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers) 
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        subset=args.subset,
        open_world=args.open_world,
        norm_family=args.norm_family,
        dataset=args.dataset
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    model, optimizer = configure_model(args, trainset)

    train = train_normal

    evaluator_val =  Evaluator(testset, model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)
    
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    scaler = torch.cuda.amp.GradScaler()

    # create model and move it to GPU with id rank
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
    

    # 💡 核心：训练循环与早停判断
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc='Current epoch', disable=not is_main_process(), ncols=100, ascii=True):
        
        trainloader.dataset.set_p(args.mixup_ratio, args.concept_shift_prob, args.obj_shift_ratio)
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        
        train(epoch, args, model, trainloader, optimizer, writer, device, scaler)
        
        # 用于多卡同步早停信号
        early_stop = torch.zeros(1).to(device)

        if is_main_process():
            if (epoch + 1) % args.eval_val_every == 0:
                with torch.no_grad():
                    # 💡 注意：修改 test 函数使其返回 stats 字典
                    stats = test(epoch, model.module if args.distributed else model, testloader, evaluator_val, writer, args, logpath, device)
                    
                    current_hm = stats['best_hm']
                    if current_hm > best_val_hm:
                        best_val_hm = current_hm
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                        print(f"\n[EarlyStopping] 性能未提升 ({epochs_since_improvement}/{patience})")
                    
                    if epochs_since_improvement >= patience:
                        early_stop += 1
        
        # 💡 同步早停信号到所有显卡
        if args.distributed:
            dist.all_reduce(early_stop, op=dist.ReduceOp.SUM)
        
        if early_stop > 0:
            if is_main_process():
                print(f"\n🛑 触发早停！已连续 {patience} 轮无提升。结束训练，进入最终测试...")
            break
            
        dist.barrier()  

    # ================= 💡 终极更新：在真实测试集上分别评估 Best HM 和 Best AUC 模型 =================
    if is_main_process():
        print("\n" + "🌟"*30)
        print("🚀 训练阶段完全结束！开始在【真实测试集(Test Set)】上进行最终评测...")
        print("🌟"*30)
        
        # 1. 强制构建真正的 Test Set 数据加载器 (phase='test')
        true_testset = dset.CompositionDataset(
            root=os.path.join(DATA_FOLDER, args.data_dir),
            phase='test',
            split=args.splitname,
            subset=args.subset,
            open_world=args.open_world,
            norm_family=args.norm_family,
            dataset=args.dataset
        )
        true_testloader = torch.utils.data.DataLoader(
            true_testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers
        )
        true_evaluator = Evaluator(true_testset, model.module)

        # ------------------- 评测 Best HM 模型 -------------------
        best_hm_path = os.path.join(logpath, 'ckpt_best_hm.t7')
        if os.path.exists(best_hm_path):
            print("\n>>> [1/2] 正在评测 【Best HM】 模型...")
            checkpoint_hm = torch.load(best_hm_path, map_location=device)
            model.module.load_state_dict(checkpoint_hm['net'])
            
            with torch.no_grad():
                test('FINAL_BestHM', model.module, true_testloader, true_evaluator, writer, args, logpath, device)
        else:
            print(f"[Warning] 找不到 Best HM 模型: {best_hm_path}")

        # ------------------- 评测 Best AUC 模型 -------------------
        best_auc_path = os.path.join(logpath, 'ckpt_best_auc.t7')
        if os.path.exists(best_auc_path):
            print("\n>>> [2/2] 正在评测 【Best AUC】 模型...")
            checkpoint_auc = torch.load(best_auc_path, map_location=device)
            model.module.load_state_dict(checkpoint_auc['net'])
            
            with torch.no_grad():
                test('FINAL_BestAUC', model.module, true_testloader, true_evaluator, writer, args, logpath, device)
        else:
            print(f"[Warning] 找不到 Best AUC 模型: {best_auc_path}")

        print("\n🎉 全部评测圆满完成！")
        print("💡 请在日志最后寻找带有 'FINAL_BestHM' 和 'FINAL_BestAUC' 的两行输出，它们就是您写进论文表格的真实成绩！")
    
    # 确保多卡同步退出 (防止非主进程提前结束导致报错)
    if args.distributed:
        dist.barrier()
    # ===============================================================


def train_normal(epoch, args, model, trainloader, optimizer, writer, device, scaler):
    '''
    Runs training for an epoch
    '''

    model.train() # Let's switch to training

    train_loss = 0.0 
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training',disable=not is_main_process(), # 仅主进程显示
        mininterval=10.0,              # 每10秒刷新一次，避免刷屏
        ncols=100,                     # 固定宽度
        ascii=True                     # 防止特殊字符乱码
        ):
        data  = [d.to(device) for d in data]

        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss, _ = model(data)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        train_loss += loss.item()

    train_loss = train_loss/len(trainloader)
    if is_main_process():
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        wandb.log({'Train/loss_total': train_loss}, step=epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, model, testloader, evaluator, writer, args, logpath, device):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    model.eval()
    model.reset_saved_pair_embeds()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing',disable=not is_main_process(), # 仅主进程显示
        mininterval=10.0,              # 每10秒刷新一次
        ncols=100,                     # 固定宽度
        ascii=True                     # 防止特殊字符乱码
        ):
        data = [d.to(device) for d in data]

        _, predictions = model(data)
        
        predictions = predictions.cpu()

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth.cpu())
        all_obj_gt.append(obj_truth.cpu())
        all_pair_gt.append(pair_truth.cpu())

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')
    
    print("Converted to CPU")

    # Calculate best unseen accuracy
    all_pred_dict = torch.cat(all_pred, dim=0)
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    print("Done Running Results")
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        wandb.log({'Val/{}'.format(key): stats[key]}, step=epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)
    # if epoch > 0 and epoch % args.save_every == 0:
    #     save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)
    
    return stats


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
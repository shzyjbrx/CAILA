import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from models.losses import MixupClassLoss
from models.clip import CLIPModel
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers import CLIPProcessor

def adjust_weights(adj, embeddings, offset):
    for idx in range(offset, adj.shape[0]):
        valid_edges = adj[idx].nonzero()[0]
        for v in valid_edges:
            adj[idx, v] = 0
            adj[v, idx] = 0
    for idx in range(adj.shape[0]):
        adj[idx, idx] = 30
    return adj

def clean_text(v):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }

    if v in custom_map:
        return custom_map[v]
    else:
        return v.lower()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CAILA(nn.Module):
    def __init__(self, dset, args):
        super(CAILA, self).__init__()
        self.args = args
        self.dset = dset
        
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)

        self.all_attrs = []
        self.all_objs = []

        for pair in self.dset.pairs:
            attr, obj = pair
            self.all_attrs.append(self.dset.attr2idx[attr])
            self.all_objs.append(self.dset.obj2idx[obj])
        
        unseen_in_vocab_idx = []
        for pair in self.dset.val_pairs + self.dset.test_pairs:
            if pair in self.dset.train_pairs:
                pass
            else:
                unseen_in_vocab_idx.append(self.dset.all_pair2idx[pair])
        self.unseen_in_vocab_idx = torch.LongTensor(unseen_in_vocab_idx)

        if self.args.train_only:
            train_idx = []
            self.all_train_attrs = []
            self.all_train_objs = []
            self.train_relations = torch.zeros((len(dset.train_pairs), len(dset.train_pairs)))
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
                attr, obj = current
                self.all_train_attrs.append(self.dset.attr2idx[attr])
                self.all_train_objs.append(self.dset.obj2idx[obj])
            self.train_idx = torch.LongTensor(train_idx)
            for i, p0 in enumerate(dset.train_pairs):
                for j, p1 in enumerate(dset.train_pairs):
                    if (i == j):
                        continue
                    attr0, obj0 = p0
                    attr1, obj1 = p1
                    if (attr0 == attr1) or (obj0 == obj1):
                        self.train_relations[i][j] = 1
            
            oov_idx = list(set([x for x in range(self.num_pairs)]) - set(train_idx) - set(unseen_in_vocab_idx))
            self.oov_idx = torch.LongTensor(oov_idx)
        else:
            self.all_train_attrs = self.all_attrs
            self.all_train_objs = self.all_objs
            self.train_idx = torch.LongTensor(list(dset.all_pair2idx.values()))

        self.open_world = args.open_world
        self.fusion_start_layer = args.fusion_start_layer
        
        config = CLIPConfig.from_pretrained("openai/{}".format(args.clip_config))
        config.text_config.reduction_factor = args.reduction_factor
        config.vision_config.reduction_factor = args.reduction_factor 

        # 文本端配置：保留 pair(组合), obj(物体), attr(属性) 的 prompt 处理
        config.text_config.track_z = False
        config.text_config.adapter_modes = ['pair', 'obj', 'attr']
        config.text_config.has_adapter = args.enable_text_adapter
        config.text_config.mixture_of_adapters = False

        # 视觉端配置：【核心修改】严格定义双分支，并关闭 MoA 混合机制
        config.vision_config.adapter_modes = ['obj', 'attr'] # 显式解耦为两个分支
        config.vision_config.track_z = False
        config.vision_config.mixture_of_adapters = False     # 关闭特征混合，保持彻底的解耦
        config.vision_config.has_adapter = True              # 强制开启视觉 Adapter
        
        # 控制 Adapter 插入的层数 (对应论文消融实验)
        # ViT-L/14 共24层 (索引为0-23)
        config.vision_config.adapter_start_layer = args.adapter_start_layer 
        config.vision_config.adapter_end_layer = 23          # 固定到最后一层
        
        config.text_config.track_z = False
        config.text_config.adapter_modes= ['pair', 'obj','attr']
        config.text_config.has_adapter = args.enable_text_adapter
        config.text_config.mixture_of_adapters = False

        config.vision_config.adapter_modes= ['obj', 'attr']
        config.vision_config.track_z = False
        config.vision_config.mixture_of_adapters = True
        config.vision_config.has_adapter = args.enable_vision_adapter
        config.vision_config.combine_latent = args.combine_latent
        config.vision_config.combine_output = args.combine_output
        config.vision_config.combination_ops = args.combination_ops
        config.vision_config.fusion_start_layer = args.fusion_start_layer
        config.vision_config.adapter_start_layer = args.adapter_start_layer
        config.vision_config.adapter_end_layer = args.adapter_end_layer
        config.vision_config.fusion_key = 'mixture'

        self.clip_model = CLIPModel(config)
        try:
            checkpoint = torch.load('./clip_ckpts/{}.pth'.format(args.clip_config), map_location='cpu')
            msg = self.clip_model.load_state_dict(checkpoint, strict=False)
        except:
            print('Could not load local clip weights, initializing from scratch')

        self.processor = CLIPProcessor.from_pretrained("openai/{}".format(args.clip_config))

        pairs = [' '.join([clean_text(t) for t in c]) for c in self.dset.pairs]
        self.pair_inputs = self.processor(text=[f"a photo of a {c}" for c in pairs], return_tensors="pt", padding=True)
        self.attr_inputs = self.processor(text=["a photo of a {} object".format(clean_text(c)) for c in self.dset.attrs], return_tensors="pt", padding=True)
        self.obj_inputs = self.processor(text=["a photo of a {}".format(clean_text(c)) for c in self.dset.objs], return_tensors="pt", padding=True)

        self.prompt_loc = 5

        self.dropout = nn.Dropout(args.img_dropout)
        
        self.attr_logit_scale = nn.Parameter(torch.ones([]) * (self.clip_model.logit_scale + math.log(20.0)))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * (self.clip_model.logit_scale + math.log(20.0)))

        self.mixup_loss = MixupClassLoss()

        self.saved_pair_embeds = None

    def reset_saved_pair_embeds(self):
        self.saved_pair_embeds = None

    def forward(self, x):
        if self.training:
            loss, pred = self.run(x)
            return loss, pred
        else:
            with torch.no_grad():
                scores = self.run(x)
            return None, scores
    
    def apply_gating_fn(self, pair_embeds, attr_embeds, obj_embeds):
        return torch.stack([pair_embeds, attr_embeds, obj_embeds], dim=-1).mean(dim=-1)

        
    def tensor_projection(self, src, dst):
        # make projection from src to dst
        
        dst_norm = F.normalize(dst, dim=-1, p=2)
        projection = torch.bmm(src.unsqueeze(1), dst.unsqueeze(-1)).squeeze(-1) * dst_norm

        return projection

    def run(self, x):
        img = x[0]
        device = img.device

        # ==================== 1. 视觉特征提取 (三分支) ====================
        # 1.1 属性和物体分支：使用 Adapter 解耦提取 (h_A, h_O)
        attr_img_feats, _ = self.clip_model.get_image_features(img, mode='attr', output_hidden_states=False)
        obj_img_feats, _ = self.clip_model.get_image_features(img, mode='obj', output_hidden_states=False)
        
        # 1.2 组合分支：因为 vision_config.adapter_modes 只有 attr 和 obj，
        # 当我们传入 mode='pair' 时，模型会绕过 Adapter，直接提取原始的全局 CLIP 视觉特征
        pair_img_feats, _ = self.clip_model.get_image_features(img, mode='pair', output_hidden_states=False)

        # L2 归一化及 Dropout
        attr_img_feats = self.dropout(F.normalize(attr_img_feats, dim=-1, p=2))
        obj_img_feats = self.dropout(F.normalize(obj_img_feats, dim=-1, p=2))
        pair_img_feats = self.dropout(F.normalize(pair_img_feats, dim=-1, p=2))

        # ==================== 2. 文本特征提取 ====================
        attr_text_embeds = self.clip_model.get_text_features(**self.attr_inputs.to(device), prompt_loc=self.prompt_loc, mode='attr')
        obj_text_embeds = self.clip_model.get_text_features(**self.obj_inputs.to(device), prompt_loc=self.prompt_loc, mode='obj')
        
        if self.training:
            # 训练阶段：仅提取当前 batch 涉及到的 pair 文本特征
            train_pair_inputs = {k: v[self.train_idx.cpu()].to(device) for k,v in self.pair_inputs.items()}
            pair_text_embeds = self.clip_model.get_text_features(**train_pair_inputs, prompt_loc=self.prompt_loc, mode='pair')
        else:
            # 推理阶段：提取所有的 pair 候选文本特征
            pair_text_embeds = self.clip_model.get_text_features(**self.pair_inputs.to(device), prompt_loc=self.prompt_loc, mode='pair')

        attr_text_embeds = F.normalize(attr_text_embeds, dim=-1, p=2).permute(1, 0)
        obj_text_embeds = F.normalize(obj_text_embeds, dim=-1, p=2).permute(1, 0)
        pair_text_embeds = F.normalize(pair_text_embeds, dim=-1, p=2).permute(1, 0)

        # ==================== 3. 计算对齐 Logits ====================
        attr_logit_scale = self.attr_logit_scale.exp()
        obj_logit_scale = self.obj_logit_scale.exp()
        pair_logit_scale = self.clip_model.logit_scale.exp() # 组合分支使用原始温度

        attr_logits = torch.matmul(attr_img_feats, attr_text_embeds) * attr_logit_scale
        obj_logits = torch.matmul(obj_img_feats, obj_text_embeds) * obj_logit_scale
        pair_logits = torch.matmul(pair_img_feats, pair_text_embeds) * pair_logit_scale

        # ==================== 4. 训练与推理逻辑 ====================
        if self.training:
            # 解析 Ground Truth，x[3] 是组合 (pair) 的标签
            attrs, objs, pairs = x[1].to(device), x[2].to(device), x[3].to(device)

            # 三分支的独立分类损失
            loss_attr = F.cross_entropy(attr_logits, attrs)
            loss_obj = F.cross_entropy(obj_logits, objs)
            loss_pair = F.cross_entropy(pair_logits, pairs)
            
            # 💡 注意：您后续的“双向条件依赖约束损失”加在这里
            # loss_constraint = ...
            
            total_loss = loss_attr + loss_obj + loss_pair
            return total_loss, None

        else:
            # 推理阶段：将三个分支的预测分数进行融合
            # 【修复】：使用 __init__ 中已经构建好的 all_attrs 和 all_objs 列表
            pair_attr_indices = torch.tensor(self.all_attrs, device=device)
            pair_obj_indices = torch.tensor(self.all_objs, device=device)
            
            # 分别提取对应的属性得分和物体得分
            scores_a = attr_logits[:, pair_attr_indices] 
            scores_o = obj_logits[:, pair_obj_indices]   
            scores_p = pair_logits 
            
            # 三分支得分融合 (相加)
            pair_scores = scores_a + scores_o + scores_p
            
            return pair_scores.cpu()
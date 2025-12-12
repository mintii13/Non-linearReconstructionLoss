import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from argparse import Namespace
import copy
import math
import os
import random
from typing import Optional
import numpy as np

from model import get_model, MODEL

# ==========================================
# 1. MFCN (Neck)
# ==========================================
class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFCN, self).__init__()
        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [in_stride / outstrides[0] for in_stride in instrides]
        self.upsample_list = nn.ModuleList([
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ])

    def forward(self, features):
        feature_list = []
        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            feature_list.append(feature_resize)
        feature_align = torch.cat(feature_list, dim=1)
        return feature_align

# ==========================================
# 2. Baseline
# ==========================================
class Baseline(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        stats_config,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.feature_masking = kwargs.get('feature_masking', None)
        self.pos_embed = build_position_embedding(pos_embed_type, feature_size, hidden_dim)
        self.save_recon = save_recon
        self.input_channel_dim = inplanes[0]
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            kwargs.get('nhead', 8), 
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False)
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if kwargs.get('normalize_before', False) else None
        self.encoder = TransformerEncoder(encoder_layer, kwargs.get('num_encoder_layers', 4), encoder_norm)
        
        decoder_layer = TransformerMemoryDecoderLayer(
            hidden_dim,
            kwargs.get('nhead', 8),
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False),
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, kwargs.get('num_decoder_layers', 4), decoder_norm, return_intermediate=False)
        
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])
        self.stats_config = stats_config
        self.activation_type = stats_config.get('activation_type', 'sigmoid').lower() if stats_config else 'sigmoid'
        
        # Xử lý K value
        k_value = 1.0
        
        k_tensor = torch.tensor([k_value], dtype=torch.float32) # Dùng tensor 1 phần tử 
        self.k_value = nn.Parameter(k_tensor, requires_grad=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        initialize_from_cfg(self, initializer)
    
    def _get_activation_fn_from_config(self, activation_type: str):
        if activation_type == 'sigmoid': return torch.sigmoid
        elif activation_type == 'tanh': return torch.tanh
        elif activation_type == 'arctan': return torch.atan
        return torch.sigmoid

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel)
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    
    def forward(self, feature_align):
        # feature_align: B x C X H x W
        feature_tokens = rearrange(feature_align, "b c h w -> (h w) b c")
        
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob)
        
        feature_tokens = self.input_proj(feature_tokens)
        
        activation_fn = self._get_activation_fn_from_config(self.activation_type)
        feature_tokens = F.layer_norm(feature_tokens, feature_tokens.shape[-1:])
        
        pos_embed = self.pos_embed(feature_tokens)
        encoded_tokens = self.encoder(feature_tokens, pos=pos_embed)
        decoded_tokens = self.decoder(encoded_tokens, encoded_tokens, pos=pos_embed)
        
        feature_rec_tokens = self.output_proj(decoded_tokens)
        is_stats_enabled = self.stats_config is not None and self.stats_config.get('enabled', False)

        if is_stats_enabled:
            # Lấy K values
            k_value = self.k_value.to(feature_align.device)
            
            # Lấy hàm activation (Sigmoid)
            activation_fn = self._get_activation_fn_from_config(self.activation_type)
            
            # Áp dụng K và Sigmoid cho Reconstruction
            feature_rec_tokens = activation_fn(feature_rec_tokens * k_value)
            feature_rec = rearrange(feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0])
            
            feature_align = activation_fn(feature_align * k_value)
            
        else:
            feature_rec = rearrange(feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0])
            feature_align = feature_align
        
        pred = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True))
        pred = self.upsample(pred)
        
        # Trả về output_dict như yêu cầu của class Baseline
        output_dict = {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }
        return output_dict


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        pos = torch.cat([pos.unsqueeze(1)] * src.size(1), dim=1)
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None: output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        output = tgt
        pos = torch.cat([pos.unsqueeze(1)] * tgt.size(1), dim=1)
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos)
            if self.return_intermediate: intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate: return torch.stack(intermediate)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before: return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerMemoryDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        if self.normalize_before: return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    if activation == "selu": return F.selu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class PositionEmbeddingSine(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False: raise ValueError("normalize should be True if scale is passed")
        if scale is None: scale = 2 * math.pi
        self.scale = scale
    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]), device=tensor.device)
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)
        return pos

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)
        j = torch.arange(self.feature_size[0], device=tensor.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            torch.cat([x_emb.unsqueeze(0)] * self.feature_size[0], dim=0),
            torch.cat([y_emb.unsqueeze(1)] * self.feature_size[1], dim=1),
        ], dim=-1).flatten(0, 1)
        return pos

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed

# ==========================================
# 4. Baseline Wrapper Class (Nối mọi thứ lại)
# ==========================================
class BaselineWrapper(nn.Module):
    def __init__(self, model_backbone, model_decoder, stats_config=None):
        super().__init__()
        self.net_backbone = get_model(model_backbone)
        self.net_merge = MFCN(
            inplanes=model_decoder['inplanes'], 
            outplanes=model_decoder['outplanes'], 
            instrides=[2, 4, 8, 16], 
            outstrides=[16]
        )
        self.net_norm = nn.LayerNorm(model_decoder['outplanes'][0], elementwise_affine=False)
        # Khởi tạo Baseline
        self.net_ad = Baseline(
            inplanes=model_decoder['outplanes'], 
            instrides=model_decoder['instrides'], 
            feature_size=model_decoder['feature_size'],
            feature_jitter=Namespace(**{'scale': 20.0, 'prob': 1.0}),
            neighbor_mask=Namespace(**{'neighbor_size': model_decoder['neighbor_size'], 'mask': [True, True, True]}),
            hidden_dim=512, 
            pos_embed_type='learned', 
            save_recon=Namespace(**{'save_dir': 'result_recon'}),
            initializer={'method': 'xavier_uniform'}, 
            stats_config=stats_config,
            nhead=8, 
            num_encoder_layers=4,
            num_decoder_layers=4, 
            dim_feedforward=1024, 
            dropout=0.1, 
            activation='relu',
            normalize_before=False
        )

        self.frozen_layers = ['net_backbone']
        self.stats_config = stats_config
    @property
    def activation_type(self):
        return self.net_ad.activation_type
    @property
    def k_value(self):
        return self.net_ad.k_value

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_backbone = self.net_backbone(imgs)
        feats_merge = self.net_merge(feats_backbone)
        # # 1. Permute
        # feats_norm = feats_merge.permute(0, 2, 3, 1) # B, H, W, C
        # # 2. Norm
        # feats_norm = self.net_norm(feats_norm)
        # # 3. Permute back
        # feats_norm = feats_norm.permute(0, 3, 1, 2) # B, C, H, W
        feats_norm = feats_merge.detach()
        
        output_dict = self.net_ad(feats_norm)
        
        # Tách dict thành tuple để trả về cho Trainer
        feature_align = output_dict['feature_align']
        feature_rec = output_dict['feature_rec']
        pred = output_dict['pred']
        
        return feature_align, feature_rec, pred

# ==========================================
# 5. Register Module
# ==========================================
@MODEL.register_module
def baseline(pretrained=False, **kwargs):
    model = BaselineWrapper(**kwargs)
    return model

def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()

def init_weights_xavier(module, method):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            if "normal" in method:
                nn.init.xavier_normal_(m.weight.data)
            elif "uniform" in method:
                nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

def init_weights_msra(module, method):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            if "normal" in method:
                nn.init.kaiming_normal_(m.weight.data, a=1)
            elif "uniform" in method:
                nn.init.kaiming_uniform_(m.weight.data, a=1)
            if m.bias is not None:
                m.bias.data.zero_()

def initialize(model, method, **kwargs):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    if method == "normal":
        init_weights_normal(model, **kwargs)
    elif "msra" in method:
        init_weights_msra(model, method)
    elif "xavier" in method:
        init_weights_xavier(model, method)

def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, "normal", std=0.01)
        return
    cfg = copy.deepcopy(cfg)
    method = cfg.pop("method")
    initialize(model, method, **cfg)
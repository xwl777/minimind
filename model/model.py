from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


from typing import Optional
import torch
import torch.nn as nn
import math

#RMSnorm implementation
class RMSNorm(nn.Module):
    # __init__ 
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepDim=True) + self.eps)
    
    # forward
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.scale

# 实现yarn
def precomput_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    # rope最初给出的频率
    freqs = 1.0 / (rope_base ** torch.arange(0, dim, 2).float() / dim)
    if rope_scaling is not None:
        orig_max, factor, beta_fast,beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 1.0),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )

        # 计算线性插值与非线性插值的临界点
        corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)

        # 计算beta
        power = torch.arange(0, dim // 2, device = freqs.device).float() / (max(dim // 2 - 1, 1))
        beta = beta_slow + (beta_fast - beta_slow) * power

        #计算scale
        scale = torch.where(torch.arange(0, dim // 2, device = freqs.device) < corr_dim,
                            (beta * factor - beta + 1.0) / (factor * beta),
                            1.0 / factor
                            )
        
        # 应用scale缩放
        freqs = freqs * scale

    # 生成位置索引
    t = torch.arange(end, device=freqs.device)

    # 外积，生成整个角度矩阵,任何同一位置相隔dim // 2维度的两个特征的角度相同
    freqs = torch.outer(t, freqs).float()
    freqs = torch.cat((freqs, freqs), dim=-1)

    # 返回cos和sin矩阵
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    
    # 获取q和k的最后一个维度大小
    dim = q.shape[-1]

    # [q1,q2]->[ -q2, q1]
    def rotate_half(x):
        x1 = x[..., :dim // 2]
        x2 = x[..., dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # 应用旋转位置编码
    q_rotated = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_rotated = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)

    return q_rotated, k_rotated
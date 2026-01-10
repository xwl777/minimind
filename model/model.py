from transformers import PretrainedConfig
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN
from typing import Optional,Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SGMindConfig(PretrainedConfig):
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
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    # factor即外推比例，即外推长度/预训练窗口长度
    if rope_scaling is not None:
        orig_max, factor, beta_fast,beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 1.0),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )

        # 仅当外推比例大于1时，才进行缩放
        if end / orig_max > 1.0:
            # 计算线性插值与非线性插值的临界点
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)

            # 计算beta
            power = torch.arange(0, dim // 2, device = freqs.device).float() / (max(dim // 2 - 1, 1))
            beta = beta_slow + (beta_fast - beta_slow) * power

            # 计算scale
            # 高频非线性插值，低频线性插值
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
    # 这个unsqueeze_dim是为了后续适配q和k的形状
    # cos和sin的形状是[seq_len, dim],而q和k的形状是[batch_size, seq_len, num_heads, head_dim]
    # 在dim=1处增加一个维度后，cos和sin的形状变为[seq_len, 1, dim]，这样就可以和q、k进行广播运算
    q_rotated = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_rotated = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)

    return q_rotated, k_rotated

# 多个Q共享一个KV，因此需要实现repeat_kv
def repeat_kv(x: torch.Tensor, n_rep: int):
    # x,即key或value
    # x的形状是[batch_size, seq_len, num_key_value_heads, head_dim]
    # 其中，num_key_value_heads是num_attention_heads的1/4
    # 重复num_key_value_heads维度n_rep次
    return x.repeat_interleave(n_rep, dim=2)

# 注意力层（GQA）
class SGMindAttention(nn.Module):
    def __init__(self, config: SGMindConfig):
        super().__init__()
        if config.num_key_value_heads is not None:
            self.num_key_value_heads = config.num_key_value_heads
        else:
            # 如果没有配置num_key_value_heads，则默认为num_attention_heads，即一个Q一个KV
            self.num_key_value_heads = config.num_attention_heads

        # 确保num_attention_heads可以被num_key_value_heads整除
        assert config.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        # 确保hidden_size可以被num_attention_heads整除
        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.n_local_heads = config.num_attention_heads
        self.n_rep = config.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # wq, wk, wv, wo
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # 注意力dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 这里还没学，先抄上
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    #forward
    def forward(self, x: torch.Tensor, 
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
                attn_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False
                ):
        # 计算 q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 把q, k, v 拆成多个头
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # 这里没有问题。minimind中这类传入的cos和sin已经是在past_position处截断的
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])

        # kv_cache处理
        if past_kv is not None:
            # 如果有past_kv，则将其与当前的k和v拼接
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        past_kv = (k, v) if use_cache else None

        # 对k和v应用repeat_kv
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        # 计算注意力得分
        # 先作转置
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        #如果使用flash attention
        # 为何是torch.all(attn_mask == 1) ？
        if self.flash and seq_len > 1 and (attn_mask is None or torch.all(attn_mask == 1)):
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(torch.full((seq_len, seq_len), float('-inf'), diagonal = 1, device = scores.device)).unsqueeze(0).unsqueeze(0)
            if attn_mask is not None:
                # 首先，根据注释可以进一步确认用户输入的 attention_mask 应该是一个 (batch, length) 的二维 tensor, 
                # 其中只包括 0 和 1 ； 此外 attention_mask 会被展开成形状为 (batch, 1, 1, length) 的四维矩阵；
                extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, seq_len, seq_len]
                extended_attn_mask = (1 - extended_attn_mask) * (-1e9)
                scores = scores + extended_attn_mask
            attn_weights = torch.softmax(scores.float(), dim=-1).type_as(q)
            attn_weights = self.attn_dropout(attn_weights)
            output = attn_weights @ v

        # 拼接头，经过out_proj输出
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # 残差dropout，但是残差这里还没加上应该
        output = self.resid_dropout(self.out_proj(output))
        return output, past_kv
    

#实现FFN
class FeedForward(nn.Module):
    # init
    # 升维全连接层
    # SiLU激活函数
    # 降维全连接层
    # 门控
    # dropout

    def __init__(self,config: SGMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = (config.hidden_size * 8) // 3
            config.intermediate_size = ((intermediate_size + 64 - 1) // 64) * 64
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.__init__
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    # forward
    def forward(self, x: torch.Tensor):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))
    

#将GQA和FFN结合成一个Transformer块
class LWXMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: SGMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_id = layer_id

        self.attn = SGMindAttention(config)
        self.ffn = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    #forward
    def forward(self,hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attn_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False
                ):
        residue = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, present_kv = self.attn(
            hidden_states,
            position_embeddings,
            attn_mask,
            past_kv,
            use_cache
        )
        hidden_states = residue + attn_output
        hidden_states = hidden_states + self.ffn(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_kv
    
class LWXMindModel(nn.Module):
    def __init__(self, config: SGMindConfig):
        super().__init__()
        # 词表大小
        self.vocab_size = config.vocab_size
        # block_layers
        self.num_hidden_layers = config.num_hidden_layers
        # embedding层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # dropout层 架构图里似乎没画
        self.dropout = nn.Dropout(config.dropout)
        # transformer blocks, i是block的层数，从0开始
        self.blocks = nn.ModuleList([LWXMindBlock(i, config) for i in range(config.num_hidden_layers)])
        # 最后的RMSNorm层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算位置编码的cos和sin
        freqs_cos, freqs_sin = precomput_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # forward
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs
                ):
        # 解包形状
        batch_size, seq_len = input_ids.shape

        # 防止传入huggingface中封装好的某些对象，而这些对象可能有layers属性，但minimind没有实现相应的处理，干脆直接扔掉
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * self.num_hidden_layers    

        # 计算start_position,用于后续位置编码
        start_position = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        position_embeddings = (
            self.freqs_cos[start_position: start_position + seq_len, :],
            self.freqs_sin[start_position: start_position + seq_len, :]
        )

        presents = []
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        for layer_idx, (block, past_kv) in enumerate(zip(self.blocks, past_key_values)):
            hidden_states, present_kv = block(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_kv,
                use_cache
            )
            presents.append(present_kv)

            #aux_loss计算暂时不实现(因为没有MOE模块)
            # aux_loss = sum(block.mlp.aux_loss for block in self.blocks if isinstance(block.mlp, MOEFeedForward))
        aux_loss = 0

        hidden_states = self.norm(hidden_states)
        
        return hidden_states, presents, aux_loss

class LWXMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SGMindConfig

    def __init__(self, config: SGMindConfig):
        self.config = config or SGMindConfig()
        super().__init__(self.config)
        self.model = LWXMindModel(self.config)
        # 输出linear层
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享.embed矩阵是一个参数巨大的矩阵，这么做可以节省显存
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs
                ):
        hidden_states, presents, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # 在prefill阶段节省计算，只计算最后logits_to_keep个token的logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output

        
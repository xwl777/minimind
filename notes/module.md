# function of different module
### author:xwl777
### update time:2025/12/25 | 2025/12/28 |2025/12/30
### last update:2025/12/30
---
1. **RMSnorm**  
   * 一个简单的归一化层，对每个句子的每个token分别归一化。只除以标准差而不减去均值，减少了计算量。
   * hyperparameter
     * dim：hidden_size的大小，即一个token映射的词向量的特征维度
     * eps：一个很小的值以防止分母为0
   * parameter
     * self.scale：一个放缩因子，大小是hidden_size，初始化为全1
   * method
     * torch.rsqrt():开根号取反
     * torch.tensor。mean(dim，keepdim)：dim=-1表示在最后一个维度取平均，keepdim=TRUE表示不减少维度（否则在某个维度取平均后该维度消失）
2. **GQA**  
   * GQA 即attention的计算。输入数据先经过RMSnorm，再线性映射得到Q、K、V，其中每4个Q共享一组K和V，给Q和K加上YaRN的旋转位置编码以及mask掩码后进行多头注意力运算。得到的结果经过一层linear映射后加上残差，构成整个GQA层。
   * RMSnorm  
     * 同***1***中描述
   * QKV
     * GQA指一个Q共享多个KV。类似的概念有MHA：一个Q即对应一个KV；MQA：所以Q共享一个KV。minimind中一个Q共享了4个K、V，因此，假设WQ将一个词向量映射到dim_q维的空间，WK和WV应该将该词向量映射到dim_q//4维的空间，并通过定义的repeat_kv方法扩展到dim_q维空间，以实现4个Q对应相同的一组KV。事实上如果使用falsh_attention，scaled_dot_product_attention方法内置了GQA，可以无序手动实现repeat_kv 
     * YaRN
       * **RoPE**  RoPE是一种相对位置编码。我们希望找到一个函数f，s.t. attention_score（q,k）= f(q,k,m-n)，其中m和n分别是q和k的位置，而显然传统的绝对位置编码给出的attention_score（q,k）= f(q,k,m,n)。  
       * **yarn** yarn是为了增加RoPE的外推能力而研究出的一种技术，值得注意的是，minimind中的yarn与标准的yarn的实现有差异，但其原理均是通过某种插值规则使得模型在外推时更好的保留信息。一般来说，高频部分捕捉局部信息，我们希望对这部分进行保留或者应用复杂的插值方式；低频部分捕捉全局信息，一般采用简单的线性插值即可。
       * 对于ROPE和YARN的进一步详细说明之后有空再补上。总之，经过了YaRN，我们使得词向量带上了相对位置信息
     * mask掩码
       * 也很简单，为了防止模型在生成某个token时提前看到后文信息而作弊，用mask掩码屏蔽Q * KT的上三角部分
   * 代码实现中的问题
     * pv_cache与RoPE不兼容问题  
        * minimind中pv_cache的实现似乎有点问题？因为原实现每次位置编码都是   
          > xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

          但是在自回归中每次传入的token显然不是前seq_len个。  

          2025/12/30更新：没有问题。看到后面会发现这里的cos和sin已经预先在past_len处截断，因此 cos[:seq_len], sin[:seq_len]完全没有问题
         
     * 掩码问题
       * 在minimind中，如果没有显式指定掩码，则可以使用flash_attention加速计算。如果显式指定了掩码，则手动计算，并且显式指定的掩码应该形如[batch_size， seq_len]，并且全为0/1。代码会将其扩展至形状[batch_size，1, 1， seq_len]并在注意力头和每个token上进行广播。我们使用代码
       ```
       scores = scores + torch.triu(torch.full((seq_len, seq_len), float('-inf'), diagonal = 1, device = scores.device)).unsqueeze(0).unsqueeze(0)
       ```
       处理因果掩码。在训练阶段和prefill阶段，输入的是词方阵，因此实现因果掩码；在推理时，输入的x是最新的token，对过去全部可见，此时seq_len为1，掩码运算结果为全0，不做任何屏蔽。
  
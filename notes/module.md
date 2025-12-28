# function of different module
### author:xwl777
### update time:2025/12/25 | 2025/12/28
### last update:2025/12/28
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
     * 正常来说经过三个矩阵乘法WQ、WK、WV即可。minimind中的特殊之处在于一个Q共享了4个K、V，因此，假设WQ将一个词向量映射到dim_q维的空间，WK和WV应该将该词向量映射到dim_q//4维的空间，并通过定义的repeat_kv方法扩展到dim_q维空间，以实现4个Q对应相同的一组KV。
     * YaRN
       * **RoPE**  RoPE是一种相对位置编码。我们希望找到一个函数f，s.t. attention_score（q,k）= f(q,k,m-n)，其中m和n分别是q和k的位置，而显然传统的绝对位置编码给出的attention_score（q,k）= f(q,k,m,n)。  
       * **yarn** yarn是为了增加RoPE的外推能力而研究出的一种技术，值得注意的是，minimind中的yarn与标准的yarn的实现有差异，但其原理均是通过某种插值规则使得模型在外推时更好的保留信息。一般来说，高频部分捕捉局部信息，我们希望对这部分进行保留或者应用复杂的插值方式；低频部分捕捉全局信息，一般采用简单的线性插值即可。
       * 对于ROPE和YARN的进一步详细说明之后有空再补上。总之，经过了YaRN，我们使得词向量带上了相对位置信息
     * mask掩码
       * 也很简单，为了防止模型在生成某个token时提前看到后文信息而作弊，用mask掩码屏蔽Q * KT的上三角部分
  
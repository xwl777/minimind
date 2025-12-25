# function of different module
### author:xwl777
### update time:2025/12/25
### last update:2025/12/25
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
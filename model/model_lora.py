import torch
from torch import  nn
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
    

# def apply_lora(model, rank=8, target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj', 'gate_proj']):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
#             print(f"Applying LoRA to module: {name}")
#             lora_module = LoRA(module.in_features, module.out_features, rank).to(model.device)
#             original_forward = module.forward
#             # 替换原始线性层的前向方法
#             def new_forward(self, x, orig_forward=original_forward, lora_forward=lora_module):
#                 return orig_forward(x) + lora_forward(x)
#             module.forward = new_forward.__get__(module)
#             # 将LoRA模块添加为子模块，便于后续加载权重
#             module.lora = lora_module
#             setattr(module, "lora", lora_module)

def apply_lora(model, rank=8, target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj', 'gate_proj']):
    # 先收集所有目标模块
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            targets.append((name, module))
    
    # 然后对每个目标模块应用LoRA
    for name, module in targets:
        lora_module = LoRA(module.in_features, module.out_features, rank).to(model.device)
        # 保存原始前向
        orig_forward = module.forward
        # 定义新前向
        def new_forward(x, orig_forward=orig_forward, lora_module=lora_module):
            return orig_forward(x) + lora_module(x)
        module.forward = new_forward
        # 将LoRA模块添加为子模块
        module.lora = lora_module
        
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path) 


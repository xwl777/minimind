import json
import torch
from torch.utils.data import Dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        # samples 是一个列表，每项是一个字典{"text": "..."}
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            # jsonl格式逐行读取
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)

        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        data = str(self.samples[index]["text"])
        encoding = self.tokenizer(
            data,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # tokenizer返回的形状是[1, max_length]
        input_ids = encoding.input_ids.squeeze()

        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask
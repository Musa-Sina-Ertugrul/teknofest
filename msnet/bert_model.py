import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline,AutoModelForCausalLM
class BertModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.model.requires_grad_(True)
        self.model.cls.predictions.decoder = torch.nn.Flatten(start_dim=2)
        self.model.cls.predictions.decoder.requires_grad_(True)
        self.linear_1 = torch.nn.Linear(768,10)
        self.linear_1.requires_grad_(True)
        self.linear_2 = torch.nn.Linear(1280,3)
        self.linear_2.requires_grad_(True)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()

    def forward(self,input_ids: torch.Tensor):
        input_ids = input_ids.to("cpu").long()
        output = self.model(input_ids).logits
        output = self.linear_1(output)
        output = self.flatten(output)
        output = self.elu(output)
        output = self.linear_2(output)
        output = self.gelu(output)
        output = self.softmax(output)
        return output
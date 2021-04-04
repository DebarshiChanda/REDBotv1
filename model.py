import torch
from transformers import AutoConfig, AutoModel


class BERTClass(torch.nn.Module):
    def __init__(self) -> object:
        super(BERTClass, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased')
        self.l1 = AutoModel.from_config(config)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

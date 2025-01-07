import torch.nn as nn

from torchvision.models import resnet34,ResNet34_Weights
from transformers import BertModel,BertTokenizer

class ResNet34Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(self.encoder.fc.in_features, embed_dim)
        self.encoder.fc = nn.Identity()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

class BertEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.encoder.config.hidden_size, embed_dim)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_embeddings)


import torch.nn as nn
import torch

class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        # Normalize embeddings
        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

        # Similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

        # Labels for contrastive loss
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Loss
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


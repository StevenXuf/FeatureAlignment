import torch.nn as nn
import torch

from torchmetrics.functional.pairwise import pairwise_cosine_similarity

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

class CustomContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5,temperature=.07):
        super().__init__()
        self.margin = margin  # Margin for negative pairs
        self.temperature=temperature

    def forward(self, x, y):
        """
        Args:
            x: Tensor of shape (batch_size, embed_dim) for image embeddings.
            y: Tensor of shape (batch_size, embed_dim) for text embeddings.
            labels: Tensor of shape (batch_size,), 1 for positive pairs, 0 for negative pairs.
        """
        cosine = pairwise_cosine_similarity(x, y)
        labels=torch.zeros_like(cosine).fill_diagonal_(1)

        positive_loss = labels*(1-cosine)
        negative_loss = (1-labels)*torch.clamp(-cosine-self.margin,min=0)

        loss = positive_loss+negative_loss

        if self.temperature:
            loss=loss/self.temperature
        
        loss=torch.sum(loss.view(1,-1),dim=-1)
        
        return loss


if __name__=='__main__':
    custom_contrastive_loss=CustomContrastiveLoss(temperature=1)
    x=torch.randn(100,256)
    y=torch.randn(100,256)
    loss=custom_contrastive_loss(x,y)
    print(loss)

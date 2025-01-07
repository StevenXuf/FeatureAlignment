import os
import torch
import torch.optim as optim
import torchvision.datasets as datasets

from tqdm import tqdm

from load_data import LoadIMDB,LoadCOCO,LoadFlickr30K

from transformers import BertTokenizer
from models import ResNet34Encoder,BertEncoder
from contrastive_loss import ContrastiveLoss

class Main():
    def __init__(self,seed,embed_dim,temp,lr,batch_size,n_epochs,dataset,ckpt_path):
        # Parameters
        self.embed_dim = embed_dim
        self.temperature = temp
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.lr=lr
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed=seed
        self.dataset=dataset
        self.ckpt_path=ckpt_path

    def load_ckpt(self,ckpt_path,modules,optimizer):
        if os.path.isfile(ckpt_path):
            ckpt=torch.load(ckpt_path)
            for module_name in modules.keys():
                modules[module_name].load_state_dict(ckpt['model_state_dict'][module_name])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            return ckpt['epoch'], ckpt['loss']
        else:
            return 0,None
    
    def save_ckpt(self,epoch,model_dict,optimizer,loss,ckpt_path):
        torch.save({
            'epoch':epoch,
            'model_state_dict':model_dict,
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss
            },ckpt_path)
        print('Model saved!')

    def train(self):
        torch.manual_seed(self.seed)
        
        # Initialize models, loss, optimizer
        vision_encoder = ResNet34Encoder(self.embed_dim).to(self.device)
        text_encoder = BertEncoder(self.embed_dim).to(self.device)
        contrastive_loss = ContrastiveLoss(self.temperature).to(self.device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        optimizer = optim.AdamW(
            list(vision_encoder.parameters()) + list(text_encoder.parameters()), lr=self.lr
        )

        # Training
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        start_epoch,_=self.load_ckpt(self.ckpt_path,{'vision_proj':vision_encoder.fc,'text_proj':text_encoder.fc},optimizer)

        for epoch in range(start_epoch,self.n_epochs):
            total_loss=.0
            for i,batch in enumerate(tqdm(dataloader)):
                images = batch['image'].to(self.device)
                encoded_captions= tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                # Forward pass
                image_embeddings = vision_encoder(images)
                text_embeddings = text_encoder(encoded_captions['input_ids'],encoded_captions['attention_mask'])
                
                # Compute loss
                loss = contrastive_loss(image_embeddings, text_embeddings)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss+=loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/(i+1):.4f}")
            print('*'*20)                
            self.save_ckpt(epoch+1,{'vision_proj':vision_encoder.fc.state_dict(),'text_proj':text_encoder.fc.state_dict()},optimizer,total_loss,self.ckpt_path)

if __name__=='__main__':
    seed=0
    dim=256
    temp=.07
    lr=1e-4
    batch_size=32
    n_epochs=100
    dataset=LoadIMDB().get_dataset()
    ckpt_dir='/data/data_fxu/ckpt_resnet34_bert/'
    os.makedirs(ckpt_dir,exist_ok=True)
    ckpt_path=ckpt_dir+'ckpt.pth'

    main=Main(seed,dim,temp,lr,batch_size,n_epochs,dataset,ckpt_path)
    main.train()

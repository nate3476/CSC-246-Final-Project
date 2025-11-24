import torch
import torch.nn as nn

class ClimbEncoder(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 256)

        #This is the "model" we will use for classification
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.fc = nn.Linear(256, 11)  

    def forward(self, input):

        x = self.embedding(input) 
        x = x.transpose(0, 1) 

        encoded = self.encoder(x)

        pooled = encoded.mean(dim=0)  

        result = self.fc(pooled)

        return result, encoded

    

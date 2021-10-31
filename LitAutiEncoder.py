import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class LitAutiEncoder(pl,LightningModule):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            #neutral, sadness, suprise, anger,fear, joy, Disgust
            nn.Linear(256, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 768))

    def forward (self, x): #x = tokenizer return value
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
        self.classifier(outputs[:,0])#first column
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import BertModel


class CEModel(pl.LightningModule):
    def __init__(self, pretrained, num_class):
        super(CEModel, self).__init__()
        self.pretrained = BertModel.from_pretrained(pretrained) # backbone
        # dimension of backbone: 64 x 512 x 512 = batch-size x max_length x feature-size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_class))

    def forward(self, x): # called to make predictions
        outputs = self.pretrained(x)
        outputs = self.classifier(outputs)

        # go thru softmax to get probability outputs
        outputs = F.softmax(outputs)
        return outputs

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
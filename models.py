import torch
import torchmetrics
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import BertModel, DistilBertModel


class CEModel(pl.LightningModule):
    def __init__(self, pretrained, num_class, loss_fn):
        super(CEModel, self).__init__()
        # backbone
        if 'distil' in pretrained:
            self.pretrained = DistilBertModel.from_pretrained(pretrained)
        else:
            self.pretrained = DBertModel.from_pretrained(pretrained)
        # dimension of backbone: 64 x 512 x 768 = batch-size x max_length x feature-size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_class))
        self.loss_fn = loss_fn

    def forward(self, x): # called to make predictions
        # encode inputs by Bert
        outputs = self.pretrained(x, return_dict=True)  # outputs shape: batch-size x text-length x feature-size
        outputs = outputs.last_hidden_state[:, 0]  # extract features of CLS token for classification

        # classify
        outputs = self.classifier(outputs)
        # go thru softmax to get probability outputs
        outputs = F.softmax(outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch

        # fast forward
        outputs = self.forward(inputs) # outputs is softmax aka probability range 0-1

        # compute loss
        loss = self.loss_fn(outputs, labels)
        acc = torchmetrics.functional.accuracy(outputs, labels)

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch

        # fast forward
        outputs = self.forward(inputs)  # outputs is softmax aka probability range 0-1

        # compute loss
        loss = self.loss_fn(outputs, labels)
        acc = torchmetrics.functional.accuracy(outputs, labels)
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, prog_bar=True, logger=True)
        return loss
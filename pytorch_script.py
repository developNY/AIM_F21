import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import Dataset
from models import CEModel


def main():
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # set hyper-parameters
    params = {'batch_size': 2, 'shuffle':True, 'num_workers': 1}
    max_epochs = 100

    # create datasets and generators
    dataset = pd.read_csv('./data/train_sent_emo.csv')
    label_dict = {k: i for i, k in enumerate(set(dataset['Emotion']))} # get label dict
    train_set = Dataset(tokenizer=tokenizer,
                          label_dict = label_dict,
                          texts=dataset['Utterance'].to_numpy(),
                          labels=dataset['Emotion'].to_numpy())
    train_generator = DataLoader(train_set, **params)

    dataset = pd.read_csv('./data/test_sent_emo.csv')
    test_set = Dataset(tokenizer=tokenizer,
                         label_dict=label_dict,
                         texts=dataset['Utterance'].to_numpy(),
                         labels=dataset['Emotion'].to_numpy())
    test_generator = DataLoader(test_set, **params)

    dataset = pd.read_csv('./data/dev_sent_emo.csv')
    dev_set = Dataset(tokenizer=tokenizer,
                        label_dict=label_dict,
                        texts=dataset['Utterance'].to_numpy(),
                        labels=dataset['Emotion'].to_numpy())
    dev_generator = DataLoader(dev_set, **params)

    # model
    model = CEModel(pretrained='bert-base-cased',
                    num_class=len(label_dict),
                    loss_fn=torch.nn.CrossEntropyLoss())

    # training
    trainer = pl.Trainer(gpus=0, precision=32)
    trainer.fit(model, train_generator, test_generator)


    # #loop over epochs
    # for epoch in range(max_epochs):
    #     #Training
    #     for local_batch, local_labels in training_generator:

    #     #validation
    #     with torch.set_grad_enabled(False):
    #         for local_batch, local_labels in validation_generator:


if __name__ == '__main__':
    main()
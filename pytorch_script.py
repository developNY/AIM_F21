import torch
from transformers import BertTokenizer
import MyDataset
from torch.utils.data import DataLoader
import LitAutiEncoder

#Data load
dataset = load_dataset('csv', data_files={'train': r'C:\Users\sksld\OneDrive\Desktop\AIM\data\train_sent_emo.csv',
                                          'test': r'C:\Users\sksld\OneDrive\Desktop\AIM\data\test_sent_emo.csv',
                                          'dev': r'C:\Users\sksld\OneDrive\Desktop\AIM\data\dev_sent_emo.csv'})
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#Parameters
params = {'batch_size':64, 'shuffle':True, 'num_workers':6}
max_epochs = 100

for i in dataset['train'].length:
    #Generator
    train_set = MyDataset(tokenizer(dataset['train']['Utterance'][i]), dataset['train']['Sr No.'][i]) #list_IDs, labels
    train_generator = DataLoader(validation_set, **params)

    test_set = MyDataset(tokenizer(dataset['test']['Utterance'][i]), dataset['test']['Sr No.'][i])
    test_generator = DataLoader(validation_set, **params)

    dev_set = MyDataset(tokenizer(dataset['dev']['Utterance'][i]), dataset['dev']['Sr No.'][i])
    dev_generator = DataLoader(validation_set, **params)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)


# #loop over epochs
# for epoch in range(max_epochs):
#     #Training
#     for local_batch, local_labels in training_generator:

#     #validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, label_dict, texts, labels):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.labels = labels  # categorize labels
        self.texts = texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        text = self.tokenizer(text, padding='max_length',
                              truncation=True)['input_ids']  # get token_ids only
        label = self.labels[index]
        label = self.label_dict[label]
        return torch.tensor(text), torch.tensor(label)

import torch
import pickle
import random
random.seed(100)


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


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, label_dict, data, min_seq=1, max_seq=5):
        super(SequenceDataset, self).__init__()
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.data, self.labels = self._process_data(data)
        self.min_seq, self.max_seq = min_seq, max_seq

    def _process_data(self, path):
        # read data
        with open(path, 'rb') as file:
            _data = pickle.load(file)

        # process data
        data, labels = [], []
        for k,v in _data.items():
            texts = list(v['Utterance'])
            texts = [self.tokenizer(x, truncation=True)['input_ids'] for x in texts]
            data.append(texts)
            labels.append(list(v['Emotion']))

        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # retrieve data and label
        _data, label = self.data[index], self.labels[index]

        # random seq
        seq_num = random.randrange(self.min_seq, self.max_seq)

        # concat and pad sequences
        data = [self.tokenizer.cls_token_id]
        idx = 0
        while idx < seq_num and idx < len(_data):
            # check length < model_max_length
            if len(data) + len(_data[idx][1:]) <= self.tokenizer.model_max_length:
                data.extend(_data[idx][1:])

            # update idx
            idx += 1

        if len(data) < self.tokenizer.model_max_length:
            pads = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(data))
            data = data[:-1] + pads + [data[-1]]

        # get label
        label = self.label_dict[label[min(idx, len(_data)-1)]]

        return torch.tensor(data), torch.tensor(label)

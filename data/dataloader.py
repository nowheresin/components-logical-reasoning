import os.path
from itertools import chain

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from text2vec import SentenceModel

from .dataset import gen_datasets


class RuleDataset(Dataset):
    def __init__(self, datasets, masks, labels):
        super().__init__()
        bert_path = 'bert_uncased' if not __name__ == '__main__' else '../bert_uncased'
        self.datas = self.gen_bert_embeding(datasets, bert_path=bert_path)
        self.rules = self.datas[:, None, :-1, :]  # (..., 1, 5, 768)
        self.target = self.datas[:, -1, :]  # (..., 768)
        self.mask = torch.tensor(masks, dtype=torch.float)  # (..., 5)
        self.label = torch.tensor(labels, dtype=torch.float)  # (..., )

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, index: int):
        return self.rules[index], self.mask[index], self.target[index], self.label[index]

    @staticmethod
    def collate_fn(data):
        # the true batch-size is sub by 4.
        rule = torch.stack([r[0] for r in data])
        rule = rule.reshape(-1, 4, 1, *rule.shape[-2:])
        mask = torch.stack([r[1] for r in data])
        mask = mask.reshape(-1, 4, 1, *mask.shape[-1:])
        target = torch.stack([r[2] for r in data])
        target = target.reshape(-1, 4, *target.shape[-1:])[..., None, :]
        label = torch.stack([r[3] for r in data])
        label = label.reshape(-1, 4)

        batch = {}
        batch['rule'] = rule
        batch['mask'] = mask
        batch['target'] = target
        batch['label'] = label
        return batch

    @staticmethod
    def gen_bert_embeding(datas, bert_path):
        assert os.path.exists(bert_path), "The bert path is wrong."
        bert_model = SentenceModel(bert_path)
        data_bert_embed = []
        for data in datas:
            embeddings = bert_model.encode(data)
            data_bert_embed.append(embeddings)

        data_bert_embed = torch.from_numpy(np.array(data_bert_embed))
        return data_bert_embed


def get_RuleDataloader(datas, example_num=100, batch_size=128, min_sample_num=None,
                       is_preprocess: bool=True, num_workers=4):
    datasets, masks, labels = gen_datasets(datas, num=example_num, min_sample_num=min_sample_num,
                                           is_preprocess=is_preprocess)
    dataset = RuleDataset(datasets, masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    return dataloader

def get_TRAIN_EVAL_DATA(ALL_DATA, num):
    assert num < len(ALL_DATA), "The number of data is wrong."
    EVAL_DATA = ALL_DATA[num]
    TRAIN_DATA = list(chain(*[DATA for i, DATA in enumerate(ALL_DATA) if i != num]))

    return TRAIN_DATA, EVAL_DATA

if __name__ == '__main__':

    """test get_TRAIN_EVAL_DATA in DEBUG mode"""
    # from database import ALL_DATA
    # get_TRAIN_EVAL_DATA(ALL_DATA=ALL_DATA, num=0)

    """test RuleDataset in DEBUG mode"""
    # from database import TRAIN_DATA
    # from dataset import gen_datasets
    # datasets, masks, labels = gen_datasets(TRAIN_DATA, num=10, min_sample_num=3)
    # dataset = RuleDataset(datasets, masks, labels)

    """test get_RuleDataloader"""
    # from database import TRAIN_DATA
    # dataloader = get_RuleDataloader(TRAIN_DATA, example_num=10, batch_size=4, min_sample_num=3)
    # for bz in dataloader:
    #     print(bz)
    #     z = bz['rule'] * bz['mask'].unsqueeze(-1).expand_as(bz['rule'])
    #     print(z)

    pass


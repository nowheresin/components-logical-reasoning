import random


def gen_state_seq(data: dict) -> list:
    num = len(data.keys())
    seq = [''] * num
    for component, features in data.items():
        state = int(features['State'][0])
        seq[state] = component
    # print(seq)
    return seq


def find_smaller_and_larger(size_seq: list, size_seq_index: list, target: int) -> list:
    sorted_list = sorted(size_seq)
    smaller = max([num for num in sorted_list if num < target], default=None)
    larger = min([num for num in sorted_list if num > target], default=None)

    smaller_index = size_seq.index(smaller) if smaller else None
    larger_index = size_seq.index(larger) if larger else None

    smaller_c = size_seq_index[smaller_index] if smaller_index is not None else None
    larger_c = size_seq_index[larger_index] if larger_index is not None else None

    return [smaller_c, larger_c]


def gen_dataset(data: dict, exclude: list=None, is_fake: bool=False, min_sample_num: int=None,
                is_preprocess: bool=True) -> tuple[list, list, list]:
    """
    A function to generate datasets, masks and labels from a data dict
    :param data: refer to database.py, such as data1 and so on
    :param exclude: exclude those components from size comparison
    :param is_fake: decide whether generate false example or not
    :param min_sample_num: decide how many characteristics to use, default(None) represents ALL
    :param is_preprocess: decide whether preprocess the data by the component relationship
                            in terms of the same characteristics or not
    :return:
    """
    state_seq = gen_state_seq(data)
    if not exclude:
        exclude = ['Base', 'Stand']
    size_seq = []
    size_seq_index = []

    datasets = []
    masks = []
    labels = []

    for component in state_seq:
        if is_fake:
            fake_components = list(set(state_seq)-{component})
        features = data[component]
        rule_data = []
        for k, v in features.items():
            if is_preprocess:
                if k == 'State' and (state := v[0]) != '0':
                    str = f'{k}_after_{state_seq[int(state) - 1]}'
                elif k == 'Size' and component not in exclude:
                    size = int(v[0])
                    if len(size_seq) > 0:
                        smaller, larger = find_smaller_and_larger(size_seq, size_seq_index, size)
                        larger_str = f'_smaller-than_{smaller}' if smaller else ''
                        smaller_str = f'_larger-than_{larger}' if larger else ''
                        str = f'{k}{larger_str}{smaller_str}'
                    else:
                        str = f'{k}_{size}'
                    size_seq.append(size)
                    size_seq_index.append(component)
                else:
                    str = f'{k}_{random.choice(v)}'
            else:
                # remove the relationship between different components
                str = f'{k}_{random.choice(v)}'
            rule_data.append(str)

        max_rule_num = len(rule_data)
        mask = [0] * max_rule_num
        min_sample_num = max_rule_num if min_sample_num is None else min_sample_num
        sample_num = random.randint(min_sample_num, max_rule_num)
        for index in random.sample(range(max_rule_num), sample_num):
            mask[index] = 1

        label = 1
        if is_fake:
            component = random.choice(fake_components)
            label = 0

        rule_data.append(component)

        datasets.append(rule_data)
        masks.append(mask)
        labels.append(label)
    return datasets, masks, labels


def gen_datasets(datas: list[dict], num: int=100, min_sample_num: int=None, is_preprocess: bool=True) -> tuple[list, list, list]:
    datasets, masks, labels = [], [], []
    # mu = 3 if min_sample_num else 1
    mu = 1
    for data in datas:
        for _ in range(num*mu):
            for is_fake in [False, True]:
                dataset, mask, label = gen_dataset(data, is_fake=is_fake, min_sample_num=min_sample_num,
                                                   is_preprocess=is_preprocess)
                datasets.extend(dataset)
                masks.extend(mask)
                labels.extend(label)
    return datasets, masks, labels


def gen_test_datasets(datas: list[dict], num: int=1, is_preprocess: bool=True) -> dict[str: list]:
    test_datasets = {}
    for _ in range(num):
        for data in datas:
            dataset, _, _ = gen_dataset(data, is_preprocess=is_preprocess)
            for rule in dataset:
                test_datasets.setdefault(component := rule[-1], [])
                test_datasets[component].append(rule[:-1])
    return test_datasets


if __name__ == '__main__':
    """test gen_dataset"""
    # from database import data1
    # x = gen_dataset(data1, min_sample_num=3, is_fake=True)
    # for _ in x:
    #     print(_)

    """test gen_datasets"""
    # from database import TRAIN_DATA
    # x = gen_datasets(TRAIN_DATA, min_sample_num=5, is_preprocess=False)
    # for _ in x:
    #     print(_)

    """test gen_test_datasets"""
    # from database import TEST_DATA
    # x = gen_test_datasets(TEST_DATA, num=10)
    # for k, v in x.items():
    #     print(k, len(v), v)

    pass

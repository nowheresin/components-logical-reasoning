import openpyxl

from data.dataloader import get_TRAIN_EVAL_DATA
from network.only_classify import only_classify
from data.dataset import gen_test_datasets
from data.database import ALL_DATA

import numpy as np
from text2vec import SentenceModel
import torch

import warnings
from loguru import logger
import sys
import os

from utils.configs import parse_arg, load_args_from_yaml

warnings.filterwarnings('ignore')
logger.remove()
logger.add(sys.stderr, level='WARNING')


def gen_masks(shape, scope: list):
    masks = np.ones(shape, dtype=int)
    mask_count = np.random.randint(*scope, size=(shape[0]))
    rows = np.arange(shape[0])
    cols = [np.random.choice(shape[1], count, replace=False) for count in mask_count]
    for row, col in zip(rows, cols):
        masks[row, col] = 0
    return masks


def eval(rule, targets, mask_scope, path='./ckpts/common/exp1/logical_reason.pt'):
    rule = np.array(rule)
    bert_model = SentenceModel('bert_uncased')
    rule_embed = torch.from_numpy(bert_model.encode(rule.reshape(-1)))
    rule_embed = rule_embed.reshape(rule.shape[0], 1, 1, rule.shape[1], rule_embed.shape[-1]).expand(-1, len(targets), -1, -1, -1)
    targets_embed = torch.from_numpy(bert_model.encode(targets))
    targets_embed = targets_embed.reshape(1, targets_embed.shape[0], 1, targets_embed.shape[-1]).expand(rule.shape[0], -1, -1, -1)
    # print(rule_embed.shape, targets_embed.shape)

    masks = gen_masks(rule.shape, mask_scope)
    mask = torch.from_numpy(masks).reshape(rule.shape[0], 1, 1, rule.shape[-1]).expand(-1, len(targets), -1, -1)

    batch = {}
    batch['rule'] = rule_embed
    batch['target'] = targets_embed
    batch['mask'] = mask

    model = only_classify()
    model.load_state_dict(torch.load(path))

    # network.check_acc_by_judge()
    indeices = model.model_eval(model, batch)
    return indeices, masks


def print_acc_err(eval_indices: list, target_indices: list, datas, targets, masks, print_debug=False):
    eval_indices, target_indices = np.array(eval_indices), np.array(target_indices)

    acc = np.mean(eval_indices == target_indices)
    # print(eval_indices, target_indices)
    # print(acc)

    diff_indices = np.where(eval_indices != target_indices)[0].tolist()

    wrong_eval_list = []
    statistics_rule_dict = {}
    statistics_component_dict = {}
    for i in diff_indices:
        if (k := eval_indices[i]) is not None:
            wrong_eval_list.append(np.array(datas[i])[masks[i].astype(bool)].tolist() + [targets[k]])
        else:
            wrong_eval_list.append(np.array(datas[i])[masks[i].astype(bool)].tolist() + ['None'])
        for item in wrong_eval_list[-1]:
            if item == wrong_eval_list[-1][-1]:
                statistics_component_dict.setdefault(item, 0)
                statistics_component_dict[item] += 1
                continue
            key = item.split('_')[0]
            statistics_rule_dict.setdefault(key, 0)
            statistics_rule_dict[key] += 1

    if print_debug == True:
        for item in wrong_eval_list:
            print(item[:-1], item[-1])

    new_statistics_rule_dict = {}
    for i in ['Material', 'Color', 'Geometry', 'Size', 'State', 'ALL']:
        new_statistics_rule_dict.setdefault(i, 0 if i not in statistics_rule_dict else statistics_rule_dict[i])
    new_statistics_rule_dict['ALL'] = len(wrong_eval_list)

    print(f"eval wrong num is {len(wrong_eval_list)}/{len(eval_indices)}, the wrong dict is {new_statistics_rule_dict}{statistics_component_dict}.")
    new_statistics_rule_values = list(map(str, new_statistics_rule_dict.values()))
    # print("\t".join(new_statistics_rule_values))
    print(f'\033[91m{targets[target_indices[0]]} pred acc: {acc*100:.2f} %.\033[0m')
    return acc, new_statistics_rule_values

def save_excel(excel_path, start_name, args, targets, ACC, ERROR_NUM):
    wb = openpyxl.Workbook()
    ws = wb.active
    # 写入数据
    ws.append([start_name])
    ws.append([str(args)])
    ws.append([''] * 2)
    ws.append(['', *targets, 'AVG_ACC'])
    ws.append(['ACC', *ACC])
    ws.append([''] * 1)
    ws.append(['ERROR_NUM', 'Material', 'Color', 'Geometry', 'Size', 'State', 'ALL'])
    [ws.append(ERR) for ERR in ERROR_NUM]
    # 保存Excel文件
    wb.save(excel_path)
    print(f"Save output into {excel_path}.")

if __name__ == '__main__':
    arg_list = ['--dir_path', './ckpts/common2_onlycls/exp1/']
    args = parse_arg(arg_list)

    path = args.dir_path  # f'./ckpts/sampling/exp1'  # common sampling
    yaml_path = os.path.join(path, 'config.yaml')
    if os.path.exists(yaml_path):
        yaml_args = load_args_from_yaml(yaml_path)
        for key, value in yaml_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        raise RuntimeError("No YAML.")

    pt_path = os.path.join(path, args.pt_name)
    print(start_name := f"{pt_path} Test Start".center(80, '-'))
    print(args)

    print_debug = False  # control the print contents

    targets = ['Front-bezel', 'Back-cover', 'Stand', 'Base', 'Circuit-boards', 'PCB-cover', 'Carrier', 'LCD-module']
    # num = 100 * 3
    num = 10 * 3
    mask_scope = [0, 3]

    # DATA = TRAIN_DATA + TEST_DATA
    # DATA = TEST_DATA
    TRAIN_DATA, EVAL_DATA = get_TRAIN_EVAL_DATA(ALL_DATA, args.eval_num)
    DATA = EVAL_DATA
    test_datasets = gen_test_datasets(DATA, num, is_preprocess=args.is_preprocess)
    all_acc = 0
    acc_list = []
    err_num_list = []
    for target_index in range(len(targets)):
        target = targets[target_index]
        datas = test_datasets[target]
        indeices, masks = eval(datas, targets, mask_scope, path=pt_path)
        acc, err_num = print_acc_err(indeices, [target_index], datas, targets, masks)
        acc_list.append(float(f"{acc * 100:.2f}"))
        err_num_list.append([target, *[int(x) for x in err_num]])
        all_acc += acc
        # break
    avg_acc = all_acc / (target_index + 1) * 100
    acc_list.append(float(f"{avg_acc:.2f}"))
    print(f'The average accuracy is {avg_acc:.2f} %.')

    excel_path = os.path.join(path, f'eval_output_num{num}_scope{mask_scope[-1]}.xlsx')
    save_excel(excel_path, start_name, args, targets, acc_list, err_num_list)

    print(f"{pt_path} Test Done".center(80, '-'))







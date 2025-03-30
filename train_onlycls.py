from network.only_classify import only_classify
from data.dataloader import get_RuleDataloader, get_TRAIN_EVAL_DATA
from data.database import ALL_DATA

from utils.configs import parse_arg, save_args_to_yaml

import os

def train(dataloader, max_epoch=200, path='./ckpts/common/exp1/logical_reason.pt'):
    model = only_classify()
    model.model_train(model, dataloader, max_epoch=max_epoch, path=path)

if __name__ == '__main__':

    arg_list = ['--dir_path', './ckpts/common2_onlycls/exp4/',
                '--min_sample_num', 'None',  # None 3
                '--example_num', '10',
                '--eval_num', '4',]  # 0, 1, 2, 3, 4
    args = parse_arg(arg_list)
    TRAIN_DATA, EVAL_DATA = get_TRAIN_EVAL_DATA(ALL_DATA, args.eval_num)

    path = args.dir_path  # f'./ckpts/common/exp'  # common sampling
    yaml_path = os.path.join(path, 'config.yaml')

    pt_path = os.path.join(path, args.pt_name)
    print(f"{pt_path} Train Start".center(80, '-'))
    print(args)
    dataloader = get_RuleDataloader(TRAIN_DATA, example_num=args.example_num, batch_size=args.batch_size,
                                    min_sample_num=args.min_sample_num, is_preprocess=args.is_preprocess)
    train(dataloader, max_epoch=args.max_epoch, path=pt_path)

    save_args_to_yaml(args, yaml_path)
    print(f"{pt_path} Train Done".center(80, '-'))

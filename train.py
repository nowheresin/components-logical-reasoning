from network.logical_reason import logical_reason
from data.dataloader import get_RuleDataloader, get_TRAIN_EVAL_DATA
from data.database import ALL_DATA

from utils.configs import parse_arg, save_args_to_yaml

import os

def train(dataloader, max_epoch=200, path='./ckpts/common/exp1/logical_reason.pt', l_logic_fac=None):
    model = logical_reason(l_logic_fac=l_logic_fac)
    model.model_train(model, dataloader, max_epoch=max_epoch, path=path)

if __name__ == '__main__':

    arg_list = ['--dir_path', './ckpts/sampling2_res_newmask_maxmask4/exp4/',  # maxmask + min_sample_num = 5
                '--min_sample_num', '1',  # None 3
                '--example_num', '10',
                '--eval_num', '4',]    # 0, 1, 2, 3, 4
                # '--is_preprocess']
    args = parse_arg(arg_list)
    TRAIN_DATA, EVAL_DATA = get_TRAIN_EVAL_DATA(ALL_DATA, args.eval_num)

    path = args.dir_path  # f'./ckpts/common/exp'  # common sampling
    yaml_path = os.path.join(path, 'config.yaml')

    pt_path = os.path.join(path, args.pt_name)
    print(f"{pt_path} Train Start".center(80, '-'))
    print(args)
    dataloader = get_RuleDataloader(TRAIN_DATA, example_num=args.example_num, batch_size=args.batch_size,
                                    min_sample_num=args.min_sample_num, is_preprocess=args.is_preprocess)
    train(dataloader, max_epoch=args.max_epoch, path=pt_path, l_logic_fac=args.l_logic_fac)

    save_args_to_yaml(args, yaml_path)
    print(f"{pt_path} Train Done".center(80, '-'))

from logical_reason.logical_reason import logical_reason
from data.dataloader import get_RuleDataloader
from data.database import TRAIN_DATA

from utils.configs import parse_arg, save_args_to_yaml

import os

def train(dataloader, max_epoch=200, path='./ckpts/common/exp1/logical_reason.pt', l_logic_fac=None):
    model = logical_reason(l_logic_fac=l_logic_fac)
    model.model_train(model, dataloader, max_epoch=max_epoch, path=path)

if __name__ == '__main__':
    args = parse_arg()

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

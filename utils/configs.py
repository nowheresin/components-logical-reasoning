import argparse
import yaml


def parse_none(value):
    if value.lower() == 'none':
        return None
    return int(value)

def parse_arg(arg_list):
    parser = argparse.ArgumentParser(description='network parameter.')
    parser.add_argument('--dir_path', type=str, help='the save/load dir path.',
                        required=True)
    parser.add_argument('--pt_name', type=str, help='the name of saved pt.',
                        default='logical_reason.pt')
    parser.add_argument('--example_num', type=int, help='the num of each example.',
                        default=100)
    parser.add_argument('--batch_size', type=int, help='batch size.',
                        default=128)
    parser.add_argument('--max_epoch', type=int, help='max epoch.',
                        default=200)
    parser.add_argument('--min_sample_num', type=parse_none, help='min sample num, None refer to ALL are chosen.',
                        default=3)  # None 3
    parser.add_argument('--l_logic_fac', type=float, help='logical loss hyper-parametric.',
                        default=0.2)
    parser.add_argument('--is_preprocess', action='store_false', help='whether preprocess by relationship.',
                        default=True)
    parser.add_argument('--eval_num', type=int, help='the one chosen to be eval.',
                        default=0)
    args = parser.parse_args(arg_list)
    return args


def save_args_to_yaml(args, filename):
    args_dict = vars(args)

    with open(filename, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def load_args_from_yaml(filename):
    with open(filename, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
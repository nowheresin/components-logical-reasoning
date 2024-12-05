import argparse
import yaml


def parse_arg():
    parser = argparse.ArgumentParser(description='model parameter.')
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
    parser.add_argument('--min_sample_num', type=int, help='min sample num, None refer to ALL are chosen.',
                        default=3)  # None 3
    parser.add_argument('--l_logic_fac', type=float, help='logical loss hyper-parametric.',
                        default=0.2)
    parser.add_argument('--is_preprocess', type=bool, help='whether preprocess by relationship.',
                        default=True)
    args = parser.parse_args()
    return args


def save_args_to_yaml(args, filename):
    args_dict = vars(args)

    with open(filename, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def load_args_from_yaml(filename):
    with open(filename, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
## Installation

- Python >= 3.11
```shell
pip install -r requriements.txt
```

## Datasets

You can check and develop your own datasets in `data/database.py`.

## Training

For training, you can simply run this code, and the model ckpt will be saved in `ckpts/common/exp`.

```shell
python train.py --dir_path ./ckpts/common/exp
```
More detailed parameters are described below (refer to `utils/configs.py`):  
`--dir_path`: `the save/load dir path.`  
`--pt_name`: `the name of saved pt.`  
`--example_num`: `the num of each example.`  
`--batch_size`: `batch size.`  
`--max_epoch`: `max epoch.`  
`--min_sample_num`: `min sample num, None refer to ALL are chosen.`  
`--l_logic_fac`: `logical loss hyper-parametric.`  
`--is_preprocess`: `whether preprocess by relationship.`  

## Testing
For testing, you can simply run this code.

```shell
python test.py --dir_path ./ckpts/common/exp
```

Make sure there are both `.pt` and `.yaml` files in `ckpts/common/exp/`.

-------
*Back to [README](README.md).*


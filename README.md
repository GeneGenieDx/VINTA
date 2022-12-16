# VINTA
VINTA -**V**isible **I**nteraction **N**etwork for **T**CR-**A**ntigen binding prediction

## Dependencies
Check `requirements.txt`.

## Folder Structure

### `bin/`
Scripts to train and run the model.

### `network/`
Model's implementation.

### `data_utils/`
Utility functions for data preprocessing.


## Example Usage

### Train a VINTA model
```shell
python bin/train_VINTA.py --data_dir data/demo --out_dir results --device cuda:0
```

### Run trained VINTA model on data
```shell
python bin/predict.py --model_config_path results/params.json \
    --model_weight_path results/VINTA.pt \
    --input_path  data/demo/random_split/test_set.tsv \
    --out_dir results/model_prediction/ \
    --name test_set
```



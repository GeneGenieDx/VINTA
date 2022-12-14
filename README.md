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
python bin/train_VINTA.py --data_dir data/training_data --out_dir results --device cuda:0
```

### Run trained VINTA model on data
```shell
python predict.py --model_config_path results/params.json \
    --model_weight_path results/cv.1/VINTA.pt \
    --input_path  data/processed_test_data/pMTnet_validation.VINTA.txt \
    --out_dir results/model_prediction/ \
    --name validation_pMTnet
```



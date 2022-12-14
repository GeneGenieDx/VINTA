#!/usr/bin/env python3
###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: whgu
## Date of creation: 11/24/2021
## Date of revision: 12/14/2022
#
##
## Description: Train and test VINTA.
#
# usage:
#   python train_VINTA.py --data_dir data/training_data --out_dir results --device cuda:0
###############################################################
import os
import sys
import json
import argparse

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)

sys.path.append(last_folder)
sys.path.append(os.path.join(last_folder, "data_utils"))

from network.VINTA import *
from dataloader import *
from train import *


# set logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
)


def train_model(
    train_loader, val_loader, test_loader, params, save_dir, device,
):
    """
    Train a VINTA model on the given dataloader.

    Params:
        - train_loader: Dataloader for training
        - val_loader: Dataloader for validation
        - test_loader : DataLoader for test
        - params: Dict
                Training parameters
        - save_dir: str
                Directory to save the training log and model's checkpoint.
        - device: torch.device
                Device on which to run the code.
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Initialize the model.
    model = VINTA(
        embedding_dim=train_loader.embedding_mat["A"].shape[0],
        tcr_padding_len=params["tcr_padding_length"],
        peptide_padding_len=params["peptide_padding_length"],
        categorical_features_dim=sum(
            [len(values) for key, values in categorical_features.items()]
        ),
        aas_cnn_channels=params["aas_cnn_channels"],
        categorical_layers_size=params["categorical_layers_size"],
        map_num=params["map_num"],
        kernel_size=params["kernel_size"],
        dropout_prob=params["dropout_prob"],
        clamp_value=params["clamp_value"],
        batch_norm=params["batch_norm"],
    )
    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    # Loss function.
    criterion = nn.BCELoss(reduction="none")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        logger=logging.getLogger(__name__),
        L1_weight=params["L1_weight"],
    )
    # Train the model and get best auc on validation data.
    best_auc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=params["epochs"],
    )
    return best_auc


def parse_input(description: str) -> argparse.ArgumentParser:
    """
    parsing input arguments
    """
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--data_dir", required=True, help="The dir where the training data is stored"
    )
    p.add_argument(
        "--peptide_split",
        required=False,
        action="store_true",
        help="The peptides in the test set will not appear in the training set",
    )
    p.add_argument("--out_dir", required=True, help="dir to store output")
    p.add_argument("--device", required=True, help="Training device")

    return p.parse_args()


def main():
    args = parse_input("Train a VINTA")
    split = "peptide_split" if args.peptide_split else "random_split"  # Split method.

    if split == "random_split":
        # Parameters for random split.
        params = {
            "L1_weight": 0.0009,
            "learning_rate": 6e-5,
            "epochs": 300,
            "batch_size": 128,
            "aas_cnn_channels": [100, 200],
            "categorical_layers_size": [20, 10],
            "map_num": 5,
            "clamp_value": None,
            "kernel_size": 3,
            "tcr_padding_length": 20,
            "peptide_padding_length": 12,
            "dropout_prob": 0,
            "batch_norm": True,
            "weight_decay": 0,
            "n_shuffle_pair": 1,
            "n_permute": 1,
            "n_bg_tcr_pair": 1,
            "n_bg_pep": 1,
        }
    else:
        # Parameters for peptide split.
        params = {
            "L1_weight": 0.0008,
            "learning_rate": 6e-5,
            "epochs": 500,
            "batch_size": 128,
            "aas_cnn_channels": [100, 200],
            "categorical_layers_size": [20, 10],
            "map_num": 5,
            "clamp_value": None,
            "kernel_size": 3,
            "tcr_padding_length": 20,
            "peptide_padding_length": 12,
            "dropout_prob": 0.2,
            "batch_norm": True,
            "weight_decay": 0,
            "n_shuffle_pair": 1,
            "n_permute": 0.5,
            "n_bg_tcr_pair": 2,
            "n_bg_pep": 0.5,
        }
    # Store the params
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(params, f)

    folds_auc = []  # The validation auc of each fold
    for fold in ["cv.1", "cv.2", "cv.3", "cv.4", "cv.5"]:
        # Load Dataset.
        logging.info("Load training set......")
        train_set = tcr_pep_dataset(
            data_path=os.path.join(args.data_dir, "tcr-peptide." + split + ".txt"),
            bg_alpha_path=os.path.join(args.data_dir, "tcr_alpha_bg_1.txt"),
            bg_beta_path=os.path.join(args.data_dir, "tcr_beta_bg_1.txt"),
            bg_pep_path=os.path.join(args.data_dir, "epitope_bg_1.txt"),
            col_split=fold,
            val_split="train",
            n_shuffle_pair=params["n_shuffle_pair"],
            n_bg_tcr_pair=params["n_bg_tcr_pair"],
            n_permute=params["n_permute"],
            n_bg_pep=params["n_bg_pep"],
            mutate_in_permute=False,
            increase_neg_weight=True if split == "peptide_split" else False,
        )
        train_loader = tcr_pep_dataloader(
            dataset=train_set,
            mode="dynamic",
            batch_size=params["batch_size"],
            device=args.device,
            shuffle=True,
            embedding="blosum",
            tcr_padding_length=params["tcr_padding_length"],
            peptide_padding_length=params["peptide_padding_length"],
        )
        logging.info("Load validation set......")
        val_set = tcr_pep_dataset(
            data_path=os.path.join(args.data_dir, "tcr-peptide." + split + ".txt"),
            bg_alpha_path=os.path.join(args.data_dir, "tcr_alpha_bg_2.txt"),
            bg_beta_path=os.path.join(args.data_dir, "tcr_beta_bg_2.txt"),
            bg_pep_path=os.path.join(args.data_dir, "epitope_bg_2.txt"),
            col_split=fold,
            val_split="val",
            n_shuffle_pair=1,
            n_bg_tcr_pair=1,
            n_permute=1,
            n_bg_pep=1,
            mutate_in_permute=True,
        )
        val_loader = tcr_pep_dataloader(
            dataset=val_set,
            mode="static",
            batch_size=params["batch_size"],
            device=args.device,
            shuffle=False,
            embedding="blosum",
            tcr_padding_length=params["tcr_padding_length"],
            peptide_padding_length=params["peptide_padding_length"],
        )
        logging.info("Load test set......")
        test_set = tcr_pep_dataset(
            data_path=os.path.join(args.data_dir, "tcr-peptide." + split + ".txt"),
            bg_alpha_path=os.path.join(args.data_dir, "tcr_alpha_bg_2.txt"),
            bg_beta_path=os.path.join(args.data_dir, "tcr_beta_bg_2.txt"),
            bg_pep_path=os.path.join(args.data_dir, "epitope_bg_2.txt"),
            col_split=fold,
            val_split="test",
            n_shuffle_pair=1,
            n_bg_tcr_pair=1,
            n_permute=1,
            n_bg_pep=1,
            mutate_in_permute=True,
        )
        test_loader = tcr_pep_dataloader(
            dataset=test_set,
            mode="static",
            batch_size=params["batch_size"],
            device=args.device,
            shuffle=False,
            embedding="blosum",
            tcr_padding_length=params["tcr_padding_length"],
            peptide_padding_length=params["peptide_padding_length"],
        )
        best_auc = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            params=params,
            save_dir=os.path.join(args.out_dir, fold),
            device=args.device,
        )
        folds_auc.append(best_auc)

    print("Average auc of five folds:", np.mean(folds_auc))
    print("Strand deviation of five folds' auc", np.std(folds_auc))


if __name__ == "__main__":
    main()

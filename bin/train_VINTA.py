#!/usr/bin/env python3
###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: whgu
## Date of creation: 11/24/2021
## Date of revision: 12/16/2022
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
    train_loader, val_loader, params, save_dir, device,
):
    """
    Train a VINTA model on the given dataloader.

    Params:
        - train_loader: Dataloader for training
        - val_loader: Dataloader for validation
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
        "--param_file",
        required=False,
        default="bin/param.json",
        help="File specifying model parameters",
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

    # Load parameters.
    with open(args.param_file, "r") as f:
        params = json.load(f).get(split)

    # Store the params
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(params, f)

    # Load training set.
    logging.info("Load training set......")
    train_set = pd.read_csv(
        os.path.join(args.data_dir, split, "train_set.tsv"), sep="\t", dtype=str
    ).fillna("")
    train_loader = tcr_pep_dataloader(
        data=train_set,
        mode="static",
        batch_size=params["batch_size"],
        device=args.device,
        shuffle=True,
        embedding="blosum",
        tcr_padding_length=params["tcr_padding_length"],
        peptide_padding_length=params["peptide_padding_length"],
    )

    # Load validation set.
    logging.info("Load validation set......")
    val_set = pd.read_csv(
        os.path.join(args.data_dir, split, "val_set.tsv"), sep="\t", dtype=str
    ).fillna("")
    val_loader = tcr_pep_dataloader(
        data=val_set,
        mode="static",
        batch_size=params["batch_size"],
        device=args.device,
        shuffle=True,
        embedding="blosum",
        tcr_padding_length=params["tcr_padding_length"],
        peptide_padding_length=params["peptide_padding_length"],
    )

    # Train a VINTA.
    best_auc = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
        save_dir=args.out_dir,
        device=args.device,
    )

    print("Best auc on validation set:", best_auc)


if __name__ == "__main__":
    main()

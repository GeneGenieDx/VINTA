###############################################################
# Copyright: GeneGenieDx Corp 2022
# Author: Minzhe Zhang
# Date of creation: 06/07/2022
# Date last modified: 12/16/2022
#
## Neoantigen
## Description: Predict TCR-peptide binding using VINTA model
#
#
## usage:
#   python predict.py \
#       --model_config_path results/params.json \
#       --model_weight_path results/VINTA.pt \
#       --input_path  data/demo/random_split/test_set.tsv \
#       --out_dir results/model_prediction/
#       --name test_set
###############################################################
#!/usr/bin/env python3
import os
import sys

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)

sys.path.append(last_folder)
sys.path.append(os.path.join(last_folder, "data_utils"))

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from network.VINTA import VINTA
from utils import valid_amino_acid_sequence, categorical_features
from dataloader import tcr_pep_dataloader


def load_model(model_config_path, model_weight_path, device) -> VINTA:
    """
    Initialize model and load trained weight.

    Params:
        model_config_path (str) : Model's configure path
        model_weight_path (str) : Path of model's weight
        device (torch.device) : Device

    Returns:
        model (VINTA) : Trained VINTA
    """
    # Initialize the model
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    model = VINTA(
        embedding_dim=20,
        tcr_padding_len=model_config["tcr_padding_length"],
        peptide_padding_len=model_config["peptide_padding_length"],
        categorical_features_dim=sum(
            [len(values) for key, values in categorical_features.items()]
        ),
        aas_cnn_channels=model_config["aas_cnn_channels"],
        categorical_layers_size=model_config["categorical_layers_size"],
        map_num=model_config["map_num"],
        kernel_size=model_config["kernel_size"],
        dropout_prob=model_config["dropout_prob"],
        clamp_value=model_config["clamp_value"],
        batch_norm=model_config["batch_norm"],
    )

    # Load weight
    model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
    model.to(device)
    model.eval()

    return model


def predict_dataloader(model, dataloader, show_bar=True):
    """
    Run model prediction on dataloader

    Params:
        dataloader (tcr_pep_dataloader): dataloader to make prediction with

    Returns:
        prediction, alpha_interaction_maps, beta_interaction_maps
    """
    prediction = []
    alpha_interaction_maps = []
    beta_interaction_maps = []
    alpha_encoded_features = []
    beta_encoded_features = []
    pep_encoded_features = []
    with torch.no_grad():
        disable = not show_bar
        for batch in tqdm(
            dataloader,
            disable=disable,
            total=dataloader.batch_count(),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            (
                pred,
                alpha_intermap,
                beta_intermap,
                (alpha_feat, beta_feat, pep_feat),
            ) = model(batch)
            # prediction
            pred = pred.squeeze().cpu().numpy().astype(float)
            prediction.extend(pred)
            # interaction map
            alpha_intermap = alpha_intermap.squeeze().cpu().numpy().astype(float)
            alpha_interaction_maps.extend(alpha_intermap)
            beta_intermap = beta_intermap.squeeze().cpu().numpy().astype(float)
            beta_interaction_maps.extend(beta_intermap)
            # encoded sequence feature
            alpha_feat = alpha_feat.squeeze().cpu().numpy().astype(float)
            beta_feat = beta_feat.squeeze().cpu().numpy().astype(float)
            pep_feat = pep_feat.squeeze().cpu().numpy().astype(float)
            alpha_encoded_features.extend(alpha_feat)
            beta_encoded_features.extend(beta_feat)
            pep_encoded_features.extend(pep_feat)

    return (
        np.array(prediction),
        np.array(alpha_interaction_maps),
        np.array(beta_interaction_maps),
        (
            np.array(alpha_encoded_features),
            np.array(beta_encoded_features),
            np.array(pep_encoded_features),
        ),
    )


def VINTA_predict(model, data, device="cpu"):
    """
    Predict with trained VINTA and background correction.

    Params: 
        model_config_path (str): Path of model's configure
        model_weight_path (str): Path of model's weight
        input_path (str): Path of the data to predict
        output_interaction_map (bool): Whether to output interaction map
    
    Returns:
        sample: data with prediction
        alpha_interaction_maps (list): alpha interaction maps of data
        beta_interaction_maps (list): beta interaction maps of data
    """
    # force mode=static, shuffle=False to keep original data and order
    sample_loader = tcr_pep_dataloader(
        data=data,
        mode="static",
        batch_size=128,
        device=device,
        shuffle=False,
        embedding="blosum",
        tcr_padding_length=model.tcr_padding_len,
        peptide_padding_length=model.peptide_padding_len,
    )
    # prediction
    (
        prediction,
        alpha_interaction_maps,
        beta_interaction_maps,
        (alpha_encoded_features, beta_encoded_features, pep_encoded_features),
    ) = predict_dataloader(model, sample_loader)
    sample = data.copy()
    sample["binding_prob"] = prediction

    return (
        sample,
        alpha_interaction_maps,
        beta_interaction_maps,
        (alpha_encoded_features, beta_encoded_features, pep_encoded_features),
    )


def squeeze_interaction_maps(inter_maps):
    """
    Squeeze 2D interaction map to 1D. Values are flatten in a manner of row by row concatenation.

    Params:
        inter_maps (list): list of 2d-array
    
    Returns:
        (pd.DataFrame): table of squeezed maps
    """
    return pd.DataFrame([m.ravel() for m in inter_maps])


def parse_input(description: str) -> argparse.ArgumentParser:
    """
    parsing input arguments
    """
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model_config_path", help="model config path")
    p.add_argument("--model_weight_path", help="path of model's weight")
    p.add_argument("--input_path", help="path of data to predict")
    p.add_argument("--out_dir", help="dir to store output")
    p.add_argument("--name", help="name of the output file")
    p.add_argument(
        "--device", required=False, default="cpu", help="cuda or cpu to run the model"
    )
    p.add_argument(
        "--vis_inter_map",
        action="store_true",
        default=False,
        help="Whether to output interaction map values",
    )

    return p.parse_args()


def main() -> None:
    global device, bg_tcr, bg_pep
    args = parse_input("Predict with trained VINTA model")
    device = args.device
    out_path = os.path.join(args.out_dir, args.name + ".pred.txt")

    # load model
    model = load_model(
        model_config_path=args.model_config_path,
        model_weight_path=args.model_weight_path,
        device=device,
    )

    # Load the dataset to predict
    feat_cols = [
        "cdr3.alpha",
        "cdr3.beta",
        "peptide",
        "mhc.i.gene",
        "mhc.i.allele",
        "v.alpha",
        "j.alpha",
        "v.beta",
        "d.beta",
        "j.beta",
    ]
    sample = pd.read_csv(args.input_path, sep="\t", dtype=str).fillna("")
    if "sample_weight" in sample.columns:
        sample = sample.drop(columns=["sample_weight"])
    sample = sample[sample.peptide.apply(lambda x: len(x) <= model.peptide_padding_len)]
    sample = sample.reset_index(drop=True)

    assert all(
        pd.Series(feat_cols).isin(sample.columns)
    ), "data must contains columns {}".format(feat_cols)
    if "label" in sample.columns:
        sample["label"] = sample["label"].astype(int)

    res, alpha_map, beta_map, _ = VINTA_predict(model=model, data=sample, device=device)
    res.to_csv(out_path, sep="\t", index=None)

    # interaction map
    if args.vis_inter_map:
        alpha_out = os.path.join(args.out_dir, args.name + ".alpha_interaction_map.txt")
        beta_out = os.path.join(args.out_dir, args.name + ".beta_interaction_map.txt")
        alpha_map = squeeze_interaction_maps(alpha_map).round(3)
        beta_map = squeeze_interaction_maps(beta_map).round(3)
        alpha_map.to_csv(alpha_out, sep="\t", index=None, header=None)
        beta_map.to_csv(beta_out, sep="\t", index=None, header=None)


if __name__ == "__main__":
    main()

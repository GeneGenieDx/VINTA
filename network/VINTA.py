###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: whgu
## Date of creation: 11/28/2021
## Date of revision: 12/1/2022
#
##
## Description: A new visible interaction network structure.
# The way to achieve that is to generate a pairwise interaction matrix (m,n) with its dimension equal to the TCR (m) and peptide (n).
# Each value in the matrix represents the interaction score between two amino acids in the TCR/peptide respectively.
# The model is compatible with alpha and alpha+beta chain input.
#
###############################################################
from typing import Dict

import torch
from torch import nn
from collections import OrderedDict

from network.utils import convolutional_layer, generate_mask


class VINTA(nn.Module):
    """
    Visible Interaction Network for Tcr-Antigen binding prediction.
    """

    def __init__(
        self,
        embedding_dim,
        tcr_padding_len,
        peptide_padding_len,
        categorical_features_dim,
        aas_cnn_channels,
        categorical_layers_size,
        map_num=3,
        kernel_size=3,
        dropout_prob=0,
        clamp_value=None,
        batch_norm=False,
    ):
        """
        Initialize the model according to the given params.

        Params:
            - embedding_dim: int,
                    Embedding dimension of each amino acids
            - tcr_padding_len: int,
                    Padding length of each TCR sequence
            - peptide_padding_len: int,
                    Padding length of each peptide sequence
            - aas_cnn_channels: list,
                    Channels of CNNs for encoding aas
            - categorical_layers_size: List,
                    Size of hidden layers of MLP for categorical features
            - categorical_features_dim: int,
                    Dimensions of categorical features
            - last_mlp_layers_size: List,
                    Size of hidden layers of last mlp
            - map_num: int,
                    The number of generated interaction maps, different maps represent different interaction properties
            - kernel_sizes: int,
                    Kernel sizes of convolutional layers for interactional maps
            - dropout_prob: float,
                    Probability of nn.Dropout()
            - clamp_value: float,
                    Upper limit of interaction map
            - batch_norm: bool,
                    Use batch norm in CNN or not

        """
        super(VINTA, self).__init__()
        self.tcr_padding_len = tcr_padding_len
        self.peptide_padding_len = peptide_padding_len
        self.dropout = nn.Dropout(dropout_prob)
        # CNN for AAs encoding.
        cnn_activation_func = nn.Tanh()
        self.aas_cnn_channels = [embedding_dim] + aas_cnn_channels
        self.tcr_encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"tcr_convolutional_{i}",
                        convolutional_layer(
                            input_channels=self.aas_cnn_channels[i],
                            num_kernel=self.aas_cnn_channels[i + 1],
                            kernel_size=3,
                            act_fn=cnn_activation_func,
                            dropout=dropout_prob,
                            batch_norm=batch_norm,
                        ),
                    )
                    for i in range(len(self.aas_cnn_channels) - 1)
                ]
            )
        )
        self.peptide_encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"peptide_convolutional_{i}",
                        convolutional_layer(
                            input_channels=self.aas_cnn_channels[i],
                            num_kernel=self.aas_cnn_channels[i + 1],
                            kernel_size=3,
                            act_fn=cnn_activation_func,
                            dropout=dropout_prob,
                            batch_norm=batch_norm,
                        ),
                    )
                    for i in range(len(self.aas_cnn_channels) - 1)
                ]
            )
        )

        # Multilayer perceptron for categorical features.
        mlp_activation_func = nn.ReLU()
        self.categorical_layers_size = [
            categorical_features_dim
        ] + categorical_layers_size
        self.mlp_categorical_features = nn.Sequential()
        for layer in range(len(self.categorical_layers_size) - 1):
            self.mlp_categorical_features.add_module(
                f"mlp_layer_{layer}",
                nn.Linear(
                    self.categorical_layers_size[layer],
                    self.categorical_layers_size[layer + 1],
                ),
            )
            self.mlp_categorical_features.add_module(
                f"act_fn_{layer}", mlp_activation_func,
            )
            self.mlp_categorical_features.add_module(
                f"drop_out_{layer}", nn.Dropout(dropout_prob)
            )
        # Modify aa with the categorical features.
        self.categorical_feature_for_alpha_chain = nn.Linear(
            self.categorical_layers_size[-1], aas_cnn_channels[-1]
        )
        self.categorical_feature_for_beta_chain = nn.Linear(
            self.categorical_layers_size[-1], aas_cnn_channels[-1]
        )
        self.categorical_feature_for_peptide_chain = nn.Linear(
            self.categorical_layers_size[-1], aas_cnn_channels[-1]
        )
        # Generate the interaction map.
        self.interaction = Interaction(aas_cnn_channels[-1], map_num)
        # Aggregate multiple interaction maps with convolution.
        self.aggregate_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=map_num,
                out_channels=1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.final_activation_func = nn.Sigmoid()
        self.bias = nn.Parameter(torch.FloatTensor([-0.5]))
        self.alpha_base_activate_value = nn.Parameter(torch.FloatTensor([0]))
        self.clamp_value = clamp_value

    def clip(self):
        self.alpha_base_activate_value.data.clamp_(min=0, max=0.2)

    def generate_interaction_map(self, data):
        """
        Generate interaction map of given TCR-peptide pairs.

        Params:
            - data: {   "alpha_chains": alpha chains in a batch,
                        "beta_chains": beta chains in a batch,
                        "peptides": peptides in a batch,
                        "categorical_features": categorical features,
                        "alpha_chain_lens": actual aas sequence length of alpha chains,
                        "beta_chain_lens": actual aas sequence length of beta chains,
                        "peptide_lens": actual aas sequence length of peptides
                    }
        Returns:
            - alpha_interaction_map: batch_size * tcr_padding_len * peptide_padding_len
            - beta_interaction_map: batch_size * tcr_padding_len * peptide_padding_len
            - encoded_alpha_chains: Tensor with the shape of (batch_size, tcr_padding_len, encoding_dim),
                    encoded alpha chains
            - encoded_beta_chains: Tensor with the shape of (batch_size, tcr_padding_len, encoding_dim),
                    encoded beta chains
            - encoded_peptides,Tensor with the shape of (batch_size, peptide_padding_len, encoding_dim),
                    encoded peptides
        """
        batch_size = data["peptides"].shape[0]
        # Encode peptides.
        encoded_peptides = self.peptide_encoder(
            data["peptides"].permute(0, 2, 1)
        ).permute(0, 2, 1)
        # Encode beta chains.
        encoded_beta_chains = self.tcr_encoder(
            data["beta_chains"].permute(0, 2, 1)
        ).permute(0, 2, 1)
        # Encode alpha chains.
        encoded_alpha_chains = self.tcr_encoder(
            data["alpha_chains"].permute(0, 2, 1)
        ).permute(0, 2, 1)
        # Encode categorical features.
        categorical_features = self.mlp_categorical_features(
            data["categorical_features"]
        )
        #  Modify aa with categorical features.
        # For peptide
        encoded_peptides = encoded_peptides * self.categorical_feature_for_peptide_chain(
            categorical_features
        ).unsqueeze(
            1
        ).expand(
            batch_size, self.peptide_padding_len, self.aas_cnn_channels[-1]
        )
        # For beta chain
        encoded_beta_chains = encoded_beta_chains * self.categorical_feature_for_beta_chain(
            categorical_features
        ).unsqueeze(
            1
        ).expand(
            batch_size, self.tcr_padding_len, self.aas_cnn_channels[-1]
        )
        # For alpha chain
        encoded_alpha_chains = encoded_alpha_chains * self.categorical_feature_for_alpha_chain(
            categorical_features
        ).unsqueeze(
            1
        ).expand(
            batch_size, self.tcr_padding_len, self.aas_cnn_channels[-1]
        )
        # Generate interaction map for beta_chain and peptide.
        beta_chains_mask = generate_mask(
            pad_tcr_len=self.tcr_padding_len,
            pad_peptide_len=self.peptide_padding_len,
            tcr_lens=data["beta_chain_lens"],
            peptide_lens=data["peptide_lens"],
        )
        beta_chain_interaction_maps = self.interaction(
            encoded_beta_chains, encoded_peptides, beta_chains_mask
        )
        # Generate interaction map for alpha_chain and peptide.
        alpha_chains_mask = generate_mask(
            pad_tcr_len=self.tcr_padding_len,
            pad_peptide_len=self.peptide_padding_len,
            tcr_lens=data["alpha_chain_lens"],
            peptide_lens=data["peptide_lens"],
        )
        alpha_chain_interaction_maps = self.interaction(
            encoded_alpha_chains, encoded_peptides, alpha_chains_mask
        )
        # Aggregate multiple interaction maps into one.
        beta_chain_aggregated_map = self.aggregate_layer(
            beta_chain_interaction_maps
        ).squeeze(1)
        alpha_chain_aggregated_map = self.aggregate_layer(
            alpha_chain_interaction_maps
        ).squeeze(1)
        return (
            alpha_chain_aggregated_map,
            alpha_chains_mask,
            beta_chain_aggregated_map,
            beta_chains_mask,
            encoded_alpha_chains,
            encoded_beta_chains,
            encoded_peptides,
        )

    def predict_from_inter_map(self, interaction_map, mask):
        """
        Predict the binding probability based on the value of interaction_map.

        Params:
            - interaction_map: Tensor with the shape of (batch_size, tcr_padding_len, peptide_padding_len),
                    Interaction map of TCR and peptide
            - mask: Tensor with the shape of (batch_size, tcr_padding_len, peptide_padding_len),
                    Mask of padding area

        Returns:
            - prediction: Tensor with the shape of (batch_size)
                    Binding probability
        """
        batch_size, tcr_len, pep_len = interaction_map.shape
        mean = torch.sum(interaction_map.view(batch_size, -1), dim=-1) / (
            torch.sum(mask.view(batch_size, -1).int(), dim=-1) + 1
        )
        binding_value = torch.relu(
            (
                interaction_map
                - mean.unsqueeze(1).unsqueeze(2).expand((batch_size, tcr_len, pep_len))
            )
            * mask
        ).view(batch_size, -1)
        prediction = torch.sum(binding_value, dim=-1) / (
            torch.sum(torch.sign(binding_value), dim=-1) + 1
        )
        return prediction

    def forward(self, data: Dict):
        """
        Params:
            - data: {   "alpha_chains": alpha chains in a batch,
                        "beta_chains": beta chains in a batch,
                        "peptides": peptides in a batch,
                        "categorical_features": categorical features,
                        "alpha_chain_lens": actual aas sequence length of alpha chains,
                        "beta_chain_lens": actual aas sequence length of beta chains,
                        "peptide_lens": actual aas sequence length of peptides
                    }

        Returns:
            - prediction: Tensor with the shape of (batch_size),
                    Binging probabilities of TCR-peptide pairs: batch_size * 1
            - alpha_interaction_map: batch_size * tcr_padding_len * peptide_padding_len
            - beta_interaction_map: batch_size * tcr_padding_len * peptide_padding_len
            - encoded_alpha_chains: Tensor with the shape of (batch_size, tcr_padding_len, encoding_dim),
                    encoded alpha chains
            - encoded_beta_chains: Tensor with the shape of (batch_size, tcr_padding_len, encoding_dim),
                    encoded beta chains
            - encoded_peptides,Tensor with the shape of (batch_size, peptide_padding_len, encoding_dim),
                    encoded peptides
        """
        (
            alpha_interaction_map,
            alpha_mask,
            beta_interaction_map,
            beta_mask,
            encoded_alpha_chains,
            encoded_beta_chains,
            encoded_peptides,
        ) = self.generate_interaction_map(data)
        # Limit the range of values in the interaction map.
        if self.clamp_value:
            alpha_interaction_map.clamp_(max=self.clamp_value)
            beta_interaction_map.clamp_(max=self.clamp_value)
        # Predict binding probability from the interaction_map.
        alpha_binding_prediction = self.predict_from_inter_map(
            alpha_interaction_map, alpha_mask
        )
        beta_binding_prediction = self.predict_from_inter_map(
            beta_interaction_map, beta_mask
        )
        # Base activation value for empty alpha chain.
        alpha_binding_prediction[
            data["alpha_chain_lens"] == 0
        ] = self.alpha_base_activate_value
        # Get the final prediction.
        prediction = self.final_activation_func(
            beta_binding_prediction
            * (
                alpha_binding_prediction
                + beta_binding_prediction
                - self.alpha_base_activate_value
            )
            + self.bias
        )
        return (
            prediction,
            alpha_interaction_map,
            beta_interaction_map,
            (encoded_alpha_chains, encoded_beta_chains, encoded_peptides),
        )


class Interaction(nn.Module):
    """
    Calculate interactions between TCR and peptide.
    """

    def __init__(self, in_dim, map_num=1):
        """
        Params:
            - in_dim: int,
                    input dimension of each amino acid
            - map_num: int,
                    Number of generated interaction maps
        """

        self.map_num = map_num
        super(Interaction, self).__init__()
        self.w = nn.Linear(in_dim, map_num * in_dim)

    def forward(self, tcrs, peptides, mask):
        """
        Params:
            - tcrs: TCRs feature (Batch_size, tcr_padding_len, input_dim)
            - peptides: Peptides feature (Batch_size, peptide_padding_len, input_dim)
        Returns:
            - Interaction maps: (Batch_size, map_num, tcr_padding_len, peptide_padding_len)
        """
        batch_size, tcr_len, input_dim = tcrs.shape
        batch_size, peptide_len, input_dim = peptides.shape
        map_num = self.map_num
        # Get hidden feature for TCRs and peptides.
        tcrs = (
            self.w(tcrs).view(batch_size, tcr_len, map_num, input_dim).transpose(1, 2)
        )
        peptides = (
            self.w(peptides)
            .view(batch_size, peptide_len, map_num, input_dim)
            .transpose(1, 2)
        )
        interaction_scores = torch.matmul(tcrs, peptides.transpose(2, 3))
        interaction_maps = interaction_scores * mask.unsqueeze(1)
        return interaction_maps

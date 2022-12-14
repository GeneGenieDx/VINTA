###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: Minzhe Zhang, whgu
## Date of creation: 11/24/2021
## Date of revision: 06/12/2022
#
## Neoantigen
## Description: DataLoader for TCR-Peptide binding prediction dataset.
#
## usage:
#   from dataloader import tcr_pep_dataloader
###############################################################

from typing import Dict
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Sampler
from dataset import tcr_pep_dataset
from data_utils.utils import (
    blosum50_20aa,
    Atchley_factors,
    convert_categorical_features,
    categorical_features,
)


# Assign an id to each amino acid and use the amino acids to tokenize amino acids sequences.
amino_acids = [letter for letter in "ARNDCEQGHILKMFPSTWYV"]  # Amino acid vocabulary.
amino_to_index = {
    amino: index for index, amino in enumerate(["PAD"] + amino_acids)
}  # Map each amino acid to id.

amino_to_index["[UNK]"] = len(amino_to_index)  # Assign a id to unknown amino acid.
tokenizer = np.vectorize(
    lambda x: amino_to_index.get(x, amino_to_index["[UNK]"]), otypes=[int]
)


class tcr_pep_dataloader(Sampler):
    """
    DataLoader for TCR-Peptide binding prediction dataset.
    """

    def __init__(
        self,
        dataset=None,
        data=None,
        mode="dynamic",
        batch_size=128,
        device="cpu",
        shuffle=True,
        embedding="blosum",
        tcr_padding_length=20,
        peptide_padding_length=12,
    ):
        """
        Read and preprocess dataset according to the given params.
        Either dataset or data should be provided.

        parameters:
            dataset (tcr_pep_dataset): wrapped up dataset for tcr-peptide pairs,
            data (pd.DataFrame): features of TCR-Peptide pairs
            mode (str): either dynmaic (resample negative each epoch) or static (same negative samples)
            batch_size (int): the number of training examples utilized in one iteration
            device (Torch.device()): "cpu" or "cuda:..."
            shuffle (bool): whether shuffle data when calling __iter__
            embedding (str): method to encode amino acid, embedding_layer or blousm, should one of the following -
                "embedding_layer", "blousm", "atchley_factors"
            tcr_padding_length (int): pad all TCR sequence to this length
            peptide_padding_length (int): pad all peptide sequence to this length
        """
        # receive either dataset or data.
        if (
            dataset is not None
            and isinstance(dataset, tcr_pep_dataset)
            and data is None
        ):
            self.dataset = dataset
            self.samples = self.dataset.samples
        elif data is not None and isinstance(data, pd.DataFrame) and dataset is None:
            self.samples = data.copy()
            cols = pd.Series(
                [
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
            )
            assert all(cols.isin(data.columns)), "Feature not complete."
            # add labels
            if "label" not in data.columns:
                self.samples["label"] = -1
            # add sample weight
            if "sample_weight" not in data.columns:
                self.samples["sample_weight"] = 1
        else:
            raise ValueError("Provide either valid dataset or data")

        # check input
        assert embedding in [
            "embedding_layer",
            "blosum",
            "atchley_factors",
        ], "Illegal embedding method."
        assert mode in ["static", "dynamic"], "Illegal mode."
        if mode == "dynamic":
            assert dataset is not None, "mode dynamic only support dataset input"
        if mode == "dynamic":
            assert shuffle == True, "mode dynamic must shuffle data."
        assert isinstance(batch_size, int) and batch_size > 0, "Illegal batch size."
        assert (
            max(self.samples["peptide"].str.len()) <= peptide_padding_length
        ), "peptide padding length too short."
        assert (
            max(self.samples["cdr3.alpha"].str.len()) <= tcr_padding_length
        ), "tcr padding length too short."
        assert (
            max(self.samples["cdr3.beta"].str.len()) <= tcr_padding_length
        ), "tcr padding length too short."

        self.embedding = embedding
        if self.embedding == "blosum":
            self.embedding_mat = blosum50_20aa
        elif self.embedding == "atchley_factors":
            self.embedding_mat = Atchley_factors
        self.mode = mode
        self.batch_size = batch_size
        self.tcr_padding_length = tcr_padding_length
        self.peptide_padding_length = peptide_padding_length
        self.device = device
        self.shuffle = shuffle

    def __len__(self):
        return len(self.samples)

    def batch_count(self):
        """
        Get the number of batches
        """
        return math.ceil(len(self.samples) / self.batch_size)

    def __iter__(self):
        self.get_batches()
        for batch in self.batches:
            yield self.preprocess_batch(batch)

    def get_batches(self):
        """
        Batchify and preprocess samples based on data mode
        """
        # resample
        if hasattr(self, "dataset"):
            if self.mode == "dynamic":
                self.dataset.resample()
            self.samples = self.dataset.samples
            self.samples["label"] = self.dataset.y
            self.samples["sample_weight"] = self.dataset.sample_weight

        # Convert categorical features into dummy variables.
        dummy_coded_var = convert_categorical_features(
            data=self.samples,
            categorical_feature_cols=list(categorical_features.keys()),
        )
        self.dummy_coded_cols = list(dummy_coded_var.columns)
        self.samples.loc[:, self.dummy_coded_cols] = dummy_coded_var

        # shuffle data
        if self.shuffle == True:
            self.samples = self.samples.sample(frac=1)

        self.batchify()

    def batchify(self) -> None:
        """
        Divide all data into batches according to the given batch size.
        """
        self.batches = []
        i = -1
        for i in range(int(np.ceil(len(self.samples) / self.batch_size)) - 1):
            self.batches.append(
                self.samples.iloc[i * self.batch_size : (i + 1) * self.batch_size, :]
            )
        self.batches.append(self.samples.iloc[(i + 1) * self.batch_size :, :])

    def preprocess_batch(self, batch) -> Dict:
        """
        Preprocess batch of data.
        1. Tokenize all TCR sequences and peptide sequences according to above amino acid vocabulary;
        2. If embedding_way==`blosum`, then encode the amino acids according to the Blosum50 scoring matrix.

        Params:
            - batch : pd.Dataframe(columns=('cdr3.alpha', 'cdr3.beta', 'peptide', ...)) with size equal to batch_size.
        Return :
                 {
                    "tcrs": preprocessed TCR sequences,
                    "peptides": preprocessed peptide sequences,
                    "tcr_lens": # The length of each TCR sequence,
                    "peptide_lens": The length of each peptide sequence,
                    "labels": The label of each TCR-Peptide pair
                 }
        """
        # Tokenize peptide and CDR3 sequences
        embedding_mat = (
            None if self.embedding == "embedding_layer" else self.embedding_mat
        )

        peptides, peptide_lens = embedding_sequence(
            sequences=batch["peptide"].values.tolist(),
            embed_mat=embedding_mat,
            padding_len=self.peptide_padding_length,
        )
        beta_chains, beta_chain_lens = embedding_sequence(
            sequences=batch["cdr3.beta"].values.tolist(),
            embed_mat=embedding_mat,
            padding_len=self.tcr_padding_length,
        )
        alpha_chains, alpha_chain_lens = embedding_sequence(
            sequences=batch["cdr3.alpha"].values.tolist(),
            embed_mat=embedding_mat,
            padding_len=self.tcr_padding_length,
        )

        return {
            # original sequence
            "alpha_chain_sequences": batch["cdr3.alpha"].values.tolist(),
            "beta_chain_sequences": batch["cdr3.beta"].values.tolist(),
            "peptide_sequences": batch["peptide"].values.tolist(),
            # embedded sequence
            "alpha_chains": torch.tensor(alpha_chains, dtype=torch.float).to(
                self.device
            ),
            "beta_chains": torch.tensor(beta_chains, dtype=torch.float).to(self.device),
            "peptides": torch.tensor(peptides, dtype=torch.float).to(self.device),
            # original sequence length
            "alpha_chain_lens": torch.tensor(alpha_chain_lens, dtype=torch.long).to(
                self.device
            ),
            "beta_chain_lens": torch.tensor(beta_chain_lens, dtype=torch.long).to(
                self.device
            ),
            "peptide_lens": torch.tensor(peptide_lens, dtype=torch.long).to(
                self.device
            ),
            # other values
            "categorical_features": torch.tensor(
                batch.loc[:, self.dummy_coded_cols].values, dtype=torch.float
            ).to(self.device),
            "sample_weight": torch.tensor(
                batch["sample_weight"].values, dtype=torch.float
            ).to(self.device),
            "labels": torch.tensor(batch["label"].values, dtype=torch.float).to(
                self.device
            ),
        }


def pad_seq(batch_seqs: np.ndarray, max_length: int) -> np.ndarray:
    """
    Pad a batch of sequences into given max_length.

    parameters:
        - batch_seqs(np.ndarray): a batch of AA sequences
        - max_length(int) : common length for padding

    returns:
        - padded sequence batch(np.ndarray): list of np.ndarrays containing padded amino acid sequences
    """

    delta = max_length - batch_seqs.shape[0]  # Number of values to be filled
    if delta <= 0:
        return batch_seqs

    return np.pad(batch_seqs, pad_width=(0, delta), constant_values=(0, 0))


def embedding_sequence(sequences, padding_len, embed_mat=None):
    """
    Embedding amino acid sequence to numeric matrix
    If the user specifies padding_length, the user specified shall prevail.
    Otherwise, we will pad all the sequences in a batch to the length of the longest sequence.
    """
    seq_lens = [len(seq) for seq in sequences]  # The length of each TCR sequence.

    if embed_mat is None:
        embed_seq = np.stack([pad_seq(seq, padding_len) for seq in sequences])
    else:
        embed_seq = embedding_with_given_matrix(sequences, embed_mat, padding_len)

    return embed_seq, seq_lens


def embedding_with_given_matrix(aa_seqs, embedding_matrix, max_seq_len):
    """
    Apply given Blosum or Atchley factors to embed amino acid sequences with padding to a max length

    parameters:
        - aa_seqs: list with AA sequences
        - embedding_matrix: dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - padded_aa_seq: list of np.ndarrays containing padded, encoded amino acid sequences
    """
    # encode sequences:
    embedded_sequences = np.array(
        [
            [
                embedding_matrix[seq[i]]
                if i < len(seq)
                else np.zeros(embedding_matrix["A"].shape[0])
                for i in range(max_seq_len)
            ]
            for seq in aa_seqs
        ]
    )  # Embedded sequences

    return embedded_sequences

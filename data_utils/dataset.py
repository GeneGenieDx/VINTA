###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: Minzhe Zhang
## Date of creation: 05/24/2022
## Date of revision: 12/14/2022
#
## Neoantigen
## Description: Dataset and Dataloader for TCR-Peptide binding
#       prediction model.
#
## usage:
#   import dataset
###############################################################

import os
import random
import pandas as pd
from torch.utils.data import Dataset
from data_utils.utils import valid_amino_acid_sequence, mutate_sequence


class tcr_pep_dataset(Dataset):
    def __init__(
        self,
        bg_alpha_path,
        bg_beta_path,
        bg_pep_path,
        data=None,
        data_path=None,
        col_split=None,
        val_split=None,
        n_shuffle_pair=1,
        n_permute=1,
        n_bg_tcr_pair=1,
        n_bg_pep=1,
        mutate_in_permute=False,
        increase_neg_weight=False,
    ):
        """
        Dataset for TCR-Peptide binding prediction model.

        Params:
            data_path (str): path of training data
            col_split (str): columns in original data contains train test splitting information
            val_split (str): value in `col_split` to select as data
            bg_tcr_path (str): path of background TCR sequence
            bg_pep_path (str): path of background peptide sequence
            n_shuffle (int): folds to generate negative samples by shuffing positive pairs
            n_permute (int): folds to generate negative samples by permute peptide sequence
            n_bg_tcr (int): folds to generate negative sample by replacing with background TCR
            n_bg_pep (int): folds to generate negative sample by replacing with background peptide
            mutate_in_permute (bool): whether introduce mutation when permutate sequence
            increase_neg_weight (bool): whether increase the sample weight of negative samples
        """
        assert (data is not None and data_path is None) or (
            data is None and data_path is not None
        ), "provide either data or data_path!"
        if data_path is not None:
            assert os.path.exists(data_path), "data file does not exists!"
        assert os.path.exists(
            bg_alpha_path
        ), "background TCR alpha file does not exists!"
        assert os.path.exists(bg_beta_path), "background TCR beta file does not exists!"
        assert os.path.exists(bg_pep_path), "background peptide file does not exists!"
        self.data = data
        self.data_path = data_path
        self.bg_alpha_path = bg_alpha_path
        self.bg_beta_path = bg_beta_path
        self.bg_pep_path = bg_pep_path
        self.n_shuffle_pair = n_shuffle_pair
        self.n_permute = n_permute
        self.n_bg_tcr_pair = n_bg_tcr_pair
        self.n_bg_pep = n_bg_pep
        self.mutate_in_permute = mutate_in_permute
        self.increase_neg_weight = increase_neg_weight

        # read data
        cols = [
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
        if self.data is None:
            self.data = pd.read_csv(self.data_path, sep="\t", dtype=str).fillna("")
        assert all(
            pd.Series(cols).isin(self.data.columns)
        ), "data must contains columns {}".format(cols)
        if col_split is not None and val_split is not None:
            self.data = self.data.loc[self.data[col_split] == val_split, :]
        if "sample_weight" not in self.data.columns:
            print("no sample_weight found, setting all to 1.")
            self.data["sample_weight"] = 1

        # Filter out invalid aas.
        self.data["peptide"] = self.data["peptide"].map(valid_amino_acid_sequence)
        self.data["cdr3.alpha"] = self.data["cdr3.alpha"].map(valid_amino_acid_sequence)
        self.data["cdr3.beta"] = self.data["cdr3.beta"].map(valid_amino_acid_sequence)
        self.data = self.data.loc[
            (self.data["cdr3.beta"] != "") & (self.data["peptide"] != "")
        ]

        # sample weight
        self.data["sample_weight"] = self.data["sample_weight"].astype(float)
        self.data["sample_weight"] = (
            self.data["sample_weight"] / self.data["sample_weight"].mean()
        )

        # sample index
        self.data = self.data[cols + ["sample_weight"]]
        self.pos_index = self.data.set_index(
            ["cdr3.alpha", "cdr3.beta", "peptide"]
        ).index

        # labels
        self.len_pos = self.data.shape[0]
        label_neg = (
            ["neg_shuffle_pair"] * int(self.len_pos * self.n_shuffle_pair)
            + ["neg_permute"] * int(self.len_pos * self.n_permute)
            + ["neg_bg_tcr_pair"] * int(self.len_pos * self.n_bg_tcr_pair)
            + ["neg_bg_pep"] * int(self.len_pos * self.n_bg_pep)
        )
        self.labels = pd.Series(["pos"] * self.len_pos + label_neg)
        self.len_neg = len(label_neg)
        self.y = (self.labels == "pos").astype(int)
        print(
            "Creating dataset: {} positive samples, {} negative samples ...".format(
                self.len_pos, self.len_neg
            ),
            flush=True,
        )

        # sequence pool
        self.tcr_alpha = pd.Series(self.data["cdr3.alpha"].unique())
        self.tcr_beta = pd.Series(self.data["cdr3.beta"].unique())
        self.pep = pd.Series(self.data["peptide"].unique())
        self.bg_alpha = pd.read_csv(self.bg_alpha_path, sep="\t", squeeze=True).map(
            valid_amino_acid_sequence
        )
        self.bg_beta = pd.read_csv(self.bg_beta_path, sep="\t", squeeze=True).map(
            valid_amino_acid_sequence
        )
        self.bg_pep = pd.read_csv(self.bg_pep_path, sep="\t", squeeze=True).map(
            valid_amino_acid_sequence
        )
        self.bg_alpha = self.bg_alpha.loc[
            ~(self.bg_alpha.isin(self.tcr_alpha) | (self.bg_alpha == ""))
        ]
        self.bg_beta = self.bg_beta.loc[
            ~(self.bg_beta.isin(self.tcr_beta) | (self.bg_beta == ""))
        ]
        self.bg_pep = self.bg_pep.loc[
            ~(self.bg_pep.isin(self.pep) | (self.bg_pep == ""))
        ]

        # sample negative data
        self.resample()

    def __len__(self):
        """
        Dataset required methods to return length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, ind):
        """
        Dataset required methods to slice dataset by index.

        Params:
            ind (int): index to slice dataset

        Return:
            sliced instance (pd.Series), instance label (int)
        """
        return self.samples.iloc[ind, :], self.y.iloc[ind], self.sample_weight.iloc[ind]

    def resample(self):
        """
        Resample negative data.
        """
        print("Sampling negative data ...", flush=True)

        # -------------- generating negative samples by shuffling ---------------- #
        # shuffle both alpha and beta (shuffle peptide)
        data_shuffle_pair = self.sample_neg_shuffling(
            df=self.data,
            nrows=int(self.len_pos * self.n_shuffle_pair),
            col_shuffle=[
                "peptide",
                "sample_weight",
            ],  # assign sample weight following peptide
            pos_index=self.pos_index,
        )
        if self.increase_neg_weight:
            data_shuffle_pair["sample_weight"] = 1

        # ------------- generating negative samples by permutation --------------- #
        data_permute = self.sample_neg_permutation(
            df=self.data,
            nrows=int(self.len_pos * self.n_permute),
            col_permute="peptide",
            pos_index=self.pos_index,
            mutate=self.mutate_in_permute,
        )
        data_permute["sample_weight"] = 1

        # -------------- generating negative samples by sampling background tcr ------------ #
        data_bg_tcr_pair = self.sample_neg_bg_tcr(
            df=self.data,
            nrows=int(self.len_pos * self.n_bg_tcr_pair),
            col_tcr=["cdr3.alpha", "cdr3.beta"],
            bg_tcr=[self.bg_alpha, self.bg_beta],
        )
        if self.increase_neg_weight:
            data_bg_tcr_pair["sample_weight"] = 1

        # ------------- generating negative samples by sampling background peptide ------------- #
        data_bg_pep = self.sample_neg_bg_peptide(
            df=self.data,
            nrows=int(self.len_pos * self.n_bg_pep),
            col_pep="peptide",
            bg_pep=self.bg_pep,
        )
        data_bg_pep["sample_weight"] = 1

        self.samples = pd.concat(
            [self.data, data_shuffle_pair, data_permute, data_bg_tcr_pair, data_bg_pep,]
        ).reset_index(drop=True)
        self.sample_weight = self.samples["sample_weight"]

    @staticmethod
    def sample_neg_shuffling(df, nrows, col_shuffle, pos_index):
        """
        Generate negative samples by shuffing positive pairs

        Params:
            df (pd.DataFrame): positive samples
            nrows (int): number of negative samples
            col_shuffle (list): column for shuffle values
            pos_index (list): list of column names as index

        Returns:
            pd.DataFrame: sampled negative sample
        """
        data_shuffle = pd.DataFrame(columns=df.columns)
        while data_shuffle.shape[0] < nrows:
            tmp = df.copy()
            tmp[col_shuffle] = tmp[col_shuffle].sample(frac=1).values
            data_shuffle = (
                pd.concat([data_shuffle, tmp])
                .drop_duplicates()
                .set_index(pos_index.names)
            )
            data_shuffle = (
                data_shuffle.loc[~data_shuffle.index.to_series().isin(pos_index), :]
                .reset_index()
                .dropna()
            )
        return data_shuffle.sample(n=nrows).reset_index(drop=True)

    @staticmethod
    def sample_neg_permutation(df, nrows, col_permute, pos_index, mutate=False):
        """
        Generate negative samples by permutation of the peptide sequence

        Params:
            df (pd.DataFrame): positive samples
            nrows (int): number of negative samples
            col_permute (str): column to permute values
            pos_index (list): list of column names as index

        Returns:
            pd.DataFrame: sampled negative sample
        """
        data_permute = pd.DataFrame(columns=df.columns)
        while data_permute.shape[0] < nrows:
            tmp = df.copy()
            # permutation
            tmp[col_permute] = tmp[col_permute].apply(
                lambda x: "".join(random.sample(x, len(x)))
            )
            # additional mutation
            if mutate:
                tmp[col_permute] = tmp[col_permute].apply(mutate_sequence)
            data_permute = (
                pd.concat([data_permute, tmp])
                .drop_duplicates()
                .set_index(pos_index.names)
            )
            data_permute = (
                data_permute.loc[~data_permute.index.to_series().isin(pos_index), :]
                .reset_index()
                .dropna()
            )
        return data_permute.sample(n=nrows).reset_index(drop=True)

    @staticmethod
    def sample_neg_bg_tcr(df, nrows, col_tcr, bg_tcr):
        """
        Generate negative samples by selecting TCR from background.

        Params:
            df (pd.DataFrame): positive samples
            nrows (int): number of negative samples
            col_tcr (list): column of tcr sequence to sample from background
            bg_tcr (list): background tcr sequence

        Returns:
            pd.DataFrame: sampled negative sample
        """
        data_bg_tcr = pd.DataFrame(columns=df.columns)
        while data_bg_tcr.shape[0] < nrows:
            tmp = df.copy()
            for col, bg in zip(col_tcr, bg_tcr):
                tmp[col] = bg.sample(n=tmp.shape[0]).values
            data_bg_tcr = pd.concat([data_bg_tcr, tmp])
        return data_bg_tcr.sample(n=nrows).reset_index(drop=True)

    @staticmethod
    def sample_neg_bg_peptide(df, nrows, col_pep, bg_pep):
        """
        Generate negative samples by selecting TCR from background.

        Params:
            df (pd.DataFrame): positive samples
            nrows (int): number of negative samples
            col_pep (list): column of peptide sequence to sample from background
            bg_pep (list): background peptide sequence

        Returns:
            pd.DataFrame: sampled negative sample
        """
        data_bg_pep = pd.DataFrame(columns=df.columns)
        while data_bg_pep.shape[0] < nrows:
            tmp = df.copy()
            bg = bg_pep.sample(n=tmp.shape[0]).values
            tmp.loc[tmp[col_pep] != "", col_pep] = bg[tmp[col_pep] != ""]
            data_bg_pep = pd.concat([data_bg_pep, tmp])
        return data_bg_pep.sample(n=nrows).reset_index(drop=True)

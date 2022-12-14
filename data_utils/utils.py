###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: Minzhe Zhang
## Date of creation: 05/23/2022
## Date of revision: 06/02/2022
#
## Neoantigen
## Description: utility functions
#
## Usage:
#   import utils
###############################################################

from typing import List
import numpy as np
import pandas as pd
import random

### ------------------------- sequence embedding data -------------------------- ###
# The Blosum50 scoring matrix for 20 amino acids.
blosum50_20aa = {
    "A": np.array(
        (5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0)
    ),
    "R": np.array(
        (-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3)
    ),
    "N": np.array(
        (-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3)
    ),
    "D": np.array(
        (-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4)
    ),
    "C": np.array(
        (-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1)
    ),
    "Q": np.array(
        (-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3)
    ),
    "E": np.array(
        (-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3)
    ),
    "G": np.array(
        (0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4)
    ),
    "H": np.array(
        (-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4)
    ),
    "I": np.array(
        (-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4)
    ),
    "L": np.array(
        (-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1)
    ),
    "K": np.array(
        (-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3)
    ),
    "M": np.array(
        (-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1)
    ),
    "F": np.array(
        (-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1)
    ),
    "P": np.array(
        (-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3)
    ),
    "S": np.array(
        (1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2)
    ),
    "T": np.array(
        (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0)
    ),
    "W": np.array(
        (-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3)
    ),
    "Y": np.array(
        (-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1)
    ),
    "V": np.array(
        (0, -3, -3, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5)
    ),
}

# The five Atchley factors of 20 amino acids, correspond loosely to polarity, secondary structure, molecular volume, codon diversity, and electrostatic charge.
Atchley_factors = {
    "A": np.array((-0.591, -1.302, -0.733, 1.57, -0.146)),
    "C": np.array((-1.343, 0.465, -0.862, -1.02, -0.255)),
    "D": np.array((1.05, 0.302, -3.656, -0.259, -3.242)),
    "E": np.array((1.357, -1.453, 1.477, 0.113, -0.837)),
    "F": np.array((-1.006, -0.59, 1.891, -0.397, 0.412)),
    "G": np.array((-0.384, 1.652, 1.33, 1.045, 2.064)),
    "H": np.array((0.336, -0.417, -1.673, -1.474, -0.078)),
    "I": np.array((-1.239, -0.547, 2.131, 0.393, 0.816)),
    "K": np.array((1.831, -0.561, 0.533, -0.277, 1.648)),
    "L": np.array((-1.019, -0.987, -1.505, 1.266, -0.912)),
    "M": np.array((-0.663, -1.524, 2.219, -1.005, 1.212)),
    "N": np.array((0.945, 0.828, 1.299, -0.169, 0.933)),
    "P": np.array((0.189, 2.081, -1.628, 0.421, -1.392)),
    "Q": np.array((0.931, -0.179, -3.005, -0.503, -1.853)),
    "R": np.array((1.538, -0.055, 1.502, 0.44, 2.897)),
    "S": np.array((-0.228, 1.399, -4.76, 0.67, -2.647)),
    "T": np.array((-0.032, 0.326, 2.213, 0.908, 1.313)),
    "V": np.array((-1.337, -0.279, -0.544, 1.242, -1.262)),
    "W": np.array((-0.595, 0.009, 0.672, -2.128, -0.184)),
    "Y": np.array((0.26, 0.83, 3.097, -0.838, 1.512)),
}

### ------------------------- sequence cleaning function -------------------------- ###
def valid_amino_acid_sequence(seq, minlen=8, maxlen=20):
    """
    Check whether the input amino acid sequence is valid:
    1. The sequence is not empty
    2. Each amino acid is in the keys of blosum
    3. Length is within minimum and maximum length

    Params:
        seq (str): input amino acid sequence
        minlen (int): minimum sequence length
        maxlen (int): maximum sequence length
    
    Return:
        seq (str): original sequence or empty string if not valid
    """
    if len(seq) < minlen or len(seq) > maxlen:
        return ""
    for aa in seq:
        if aa not in blosum50_20aa:
            return ""
    return seq


def valid_tcr_sequence(seq):
    """
    Check whether the input amino acid sequence is valid:
    1. the sequence starts with C and ends with F (or W)
    2. If not, append to make it valid

    Params:
        seq (str): cdr3 sequence
    
    Return:
        newseq (str): original or corrected sequence if not valid
    """
    if seq == "":
        return ""
    newseq = seq[:]
    if seq[0] != "C":  # start with C
        newseq = "C" + newseq
    if seq[-1] not in ["F", "W"]:  # ends with F or W
        newseq = newseq + "F"
    if seq[-2:] == "FF":  # chop double F to keep only one F
        newseq = newseq[:-1]
    return newseq


def mutate_sequence(seq):
    """
    Introduce random mutation to sequence.

    Params:
        seq: original sequence

    Returns:
        mutated sequence
    """
    aa = "ARNDCEQGHILKMFPSTWYV"
    ind = random.randint(0, len(seq) - 1)
    r = random.sample(aa, 1)[0]
    if seq[ind] != r:
        new = seq[0:ind] + r + seq[ind + 1:]
        return new
    else:
        mutate_sequence(seq)


### ------------------------- model feature processing -------------------------- ###
categorical_features = {
    "mhc.i.gene": ["", "HLA-A", "HLA-B", "HLA-C"],
    "mhc.i.allele": [
        "", "01", "02", "03", "04", "05", "06", "07", "08", "11", "12", "13", "14", "15", "16", "18",
        "23", "24", "26", "27", "29", "32", "34", "35", "37", "39", "40", "42", "44", "46", "51", "52", "57", "66", "68"
    ],
    "v.alpha": [
        "", "TRAV1", "TRAV10", "TRAV12", "TRAV13", "TRAV14", "TRAV16", "TRAV17", "TRAV19", 
        "TRAV2", "TRAV20", "TRAV21", "TRAV22", "TRAV23", "TRAV24", "TRAV25", "TRAV26", "TRAV27", "TRAV29",
        "TRAV3", "TRAV30", "TRAV34", "TRAV35", "TRAV36", "TRAV38", "TRAV39",
        "TRAV4", "TRAV41", "TRAV5", "TRAV6", "TRAV8", "TRAV9"
    ],
    "j.alpha": [
        "", "TRAJ10", "TRAJ11", "TRAJ12", "TRAJ13", "TRAJ15", "TRAJ17", "TRAJ18",
        "TRAJ20", "TRAJ21", "TRAJ22", "TRAJ23", "TRAJ24", "TRAJ26", "TRAJ27", "TRAJ28", "TRAJ29",
        "TRAJ3", "TRAJ30", "TRAJ31", "TRAJ32", "TRAJ33", "TRAJ34", "TRAJ36", "TRAJ37", "TRAJ38", "TRAJ39",
        "TRAJ4", "TRAJ40", "TRAJ41", "TRAJ42", "TRAJ43", "TRAJ44", "TRAJ45", "TRAJ47", "TRAJ48", "TRAJ49",
        "TRAJ5", "TRAJ50", "TRAJ52", "TRAJ53", "TRAJ54", "TRAJ56", "TRAJ57", 
        "TRAJ6", "TRAJ7", "TRAJ8", "TRAJ9"
    ],
    "v.beta": [
        "", "TRBV1", "TRBV10", "TRBV11", "TRBV12", "TRBV13", "TRBV14", "TRBV15", "TRBV16", "TRBV18", "TRBV19",
        "TRBV2", "TRBV20", "TRBV21", "TRBV23", "TRBV24", "TRBV25", "TRBV27", "TRBV28", "TRBV29",
        "TRBV3", "TRBV30", "TRBV4", "TRBV5", "TRBV6", "TRBV7", "TRBV9"
    ],
    "d.beta": ["", "TRBD1", "TRBD2"],
    "j.beta": ["", "TRBJ1", "TRBJ2"]
}

def dummy_variable(feat, vals, prefix="", na_val=""):
    """
    Convert a categorical feature into dummy variables.

    Params:
        feat (pd.Series) : Data of which to get dummy variables
        vals (list): predefined varibles space
        prefix (str): prefix to add before variables
        na_val (str): nan value to replace when value not found in varibles space
    
    Returns:
        Converted dummy variables.
    """
    # replace nan value
    s = feat.copy()
    s.loc[~s.isin(vals)] = na_val

    # dummy variables
    dummy = pd.DataFrame(index=s.index)
    for v in vals:
        dummy[prefix+"_"+v] = (s == v).astype(int)
    
    return dummy


def convert_categorical_features(
    data: pd.DataFrame, categorical_feature_cols: List = None
):
    """
    Convert categorical features into dummy variables.

    Params:
        data (pd.DataFrame) : Data of which to get dummy variables.
        categorical_feature_cols (List): Columns of categorical features.

    Returns:
        converted_features (pd.DataFrame) : Converted categorical features.
    """
    # default use all categorical features
    if not categorical_feature_cols:
        categorical_feature_cols = categorical_features.keys()
    
    # convert 
    dummy_coded_data = [
        dummy_variable(feat=data[col], vals=categorical_features[col], prefix=col) for col in categorical_feature_cols
    ]
    converted_features = pd.concat(dummy_coded_data, axis=1)

    return converted_features

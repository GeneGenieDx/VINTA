import torch
from torch import nn
from collections import OrderedDict


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    max_pooling=False,
    dropout=0.0,
    input_channels=1,
    padding=1,
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "convolve",
                    torch.nn.Conv1d(
                        input_channels, num_kernel, kernel_size, padding=padding,
                    ),
                ),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
                ("maxpool", nn.AdaptiveMaxPool1d(1) if max_pooling else nn.Identity()),
                (
                    "batch_norm",
                    nn.BatchNorm1d(num_kernel) if batch_norm else nn.Identity(),
                ),
            ]
        )
    )


def generate_mask(pad_tcr_len, pad_peptide_len, tcr_lens, peptide_lens):
    """
    Generate mask for interaction map according actual aas sequence length.
    Params:
        - pad_tcr_len: Int,
            Padding length of each TCR sequence.
        - pad_peptide_len: Int,
            Padding length of each Peptide sequence.
        - tcr_lens: batch_size*1,
                Actual aas sequence length of TCRs.
        - peptide_lens: batch_size*1,
                Actual aas sequence length of Peptides.
    Return:
        - mask: batch_size*pad_tcr_len*pad_peptide_len
    """

    batch_size = tcr_lens.shape[0]
    tcr_mask = (
        (
            torch.arange(0, pad_tcr_len, device=tcr_lens.device)
            .unsqueeze(0)
            .expand(batch_size, pad_tcr_len)
            .lt(tcr_lens.unsqueeze(1))
        )
        .unsqueeze(2)
        .repeat(1, 1, pad_peptide_len)
    )
    peptide_mask = (
        (
            torch.arange(0, pad_peptide_len, device=tcr_lens.device)
            .unsqueeze(0)
            .expand(batch_size, pad_peptide_len)
            .lt(peptide_lens.unsqueeze(1))
        )
        .unsqueeze(1)
        .repeat(1, pad_tcr_len, 1)
    )
    mask = tcr_mask & peptide_mask

    return mask

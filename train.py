###############################################################
## Copyright: GeneGenieDx Corp 2021
## Author: whgu
## Date of creation: 11/24/2021
## Date of revision: 12/12/2022
#
##
## Description: Class to handle training model.
#
# usage:
#   import train.py
###############################################################
import os
import torch
import logging
from timeit import default_timer

from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import trange


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        save_dir,
        logger=logging.getLogger(__name__),
        L1_weight=0,
        is_progress_bar=True,
    ):
        """
        Class to handle training of model.

        Params:
            - model: initialized VINTA
            - optimizer: torch.optim.Optimizer
            - criterion: Loss function.
            - device: torch.device
                    Device on which to run the code.
            - save_dir : str,
                    Directory for saving logs and model.
            - logger: logging.Logger, optional
                    Logger.
            - L1_weight: float
                    Weight for L1 regularization of the interaction map in loss.
            - is_progress_bar: bool, optional
                    Whether to use a progress bar for training.
        """
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.L1_weight = L1_weight
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.train_logger = LossesLogger(os.path.join(self.save_dir, "train.log"))
        self.logger.info("Training Device: {}".format(self.device))

    def train(
        self, train_loader, val_loader, test_loader, epochs=100,
    ):
        """
        Trains the model.

        Params:
            - train_loader : DataLoader for training
            - val_loader : DataLoader for validation
            - test_loader : DataLoader for test
            - epochs : int, optional
                    Number of epochs to train the model for.

        Returns:
            - best_auc : float,
                Best auc on validation data.
        """
        start = default_timer()
        best_auc = 0.0

        for epoch in range(epochs):
            self.model.train()
            (
                train_loss,
                train_cls_loss,
                train_cls_loss_with_weight,
                train_L1_loss,
                train_auc,
                train_ap,
            ) = self._train_epoch(train_loader, epoch)
            # Get metric on validation data.
            (val_loss, val_auc, val_ap,) = self._validate_epoch(val_loader)
            # Get metric on test data.
            (test_loss, test_auc, test_ap,) = self._validate_epoch(test_loader)

            self.logger.info(
                "Epoch: {} Train loss : {:.6f}, Train_cls_loss : {:.6f}, Train L1_loss : {:.6f}, Train_auc: {:.6f}, Test_loss : {:.6f}, Test_auc : {:.6f}".format(
                    epoch + 1,
                    train_loss,
                    train_cls_loss,
                    train_L1_loss,
                    train_auc,
                    test_loss,
                    test_auc,
                )
            )
            self.logger.info("-" * 100)
            self.train_logger.log(
                epoch,
                train_loss,
                train_cls_loss,
                train_cls_loss_with_weight,
                train_L1_loss,
                train_auc,
                train_ap,
                val_loss,
                val_auc,
                val_ap,
                test_loss,
                test_auc,
                test_ap,
            )
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    self.model.state_dict(), os.path.join(self.save_dir, "VINTA.pt"),
                )
        delta_time = (default_timer() - start) / 60
        self.logger.info(
            "Finished training after {:.1f} min, Best AUC on validation set : {:.6f}".format(
                delta_time, best_auc
            )
        )
        return best_auc

    def _train_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.

        Params:
            - train_loader : DataLoader for training
            - epoch : int, Epoch number

        Returns:
            - train_loss : float
                Mean loss of training data
            - train_cls_loss: float
                Mean classification loss of training data
            - train_L1_loss: float
                Mean L1 loss of training data
            - train_auc : float
                AUC of model's prediction on training data
            - train_precision : float
                Precision of model's prediction on training data
        """
        train_cls_loss = 0.0  # Classification loss
        train_cls_loss_with_weight = (
            0.0  # Classification loss after multiplying the sample weight
        )
        train_L1_loss = 0.0  # L1 loss
        train_targets = []  # Labels of training instances
        train_preds = []  # Model's predictions of training instances
        kwargs = dict(
            desc="Epoch {}".format(epoch + 1),
            leave=False,
            disable=not self.is_progress_bar,
        )
        with trange(train_loader.batch_count(), **kwargs) as t:
            for data in train_loader:
                (
                    iter_prediction,
                    iter_target,
                    iter_cls_loss,
                    iter_cls_loss_with_weight,
                    iter_L1_loss,
                ) = self._train_iteration(data)

                train_cls_loss += iter_cls_loss.item()
                train_cls_loss_with_weight += iter_cls_loss_with_weight.item()
                train_L1_loss += iter_L1_loss.item()
                train_targets.extend(iter_target.cpu().tolist())
                train_preds.extend(iter_prediction.cpu().detach().tolist())
                t.update()

        train_loss = train_cls_loss_with_weight + train_L1_loss
        # Calculate AUC and precision
        train_auc = roc_auc_score(train_targets, train_preds)
        train_precision = average_precision_score(train_targets, train_preds)
        return (
            train_loss / train_loader.batch_count(),
            train_cls_loss / train_loader.batch_count(),
            train_cls_loss_with_weight / train_loader.batch_count(),
            train_L1_loss / train_loader.batch_count(),
            train_auc,
            train_precision,
        )

    def _validate_epoch(self, val_loader):
        """
        Test the model on the validation set.

        Params:
            - val_loader: DataLoader for validation

        Returns:
            - val_loss : float
                Mean loss of validation data
            - val_auc : float
                AUC of model's prediction on validation data
            - val_precision : float
                Precision of model's prediction on validation data
        """
        # After each epoch, test the model on the validation set.
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_targets = []  # Labels of testing instances
            val_preds = []  # Model's prediction of testing instances
            for data in val_loader:
                torch.cuda.empty_cache()
                prediction, alpha_interaction_map, beta_interaction_map, _ = self.model(
                    data
                )
                val_preds.extend(prediction.cpu().detach().tolist())
                val_targets.extend(data["labels"].cpu().tolist())
                target = data["labels"]
                loss = self.criterion(prediction.squeeze(), target.squeeze()).mean()
                val_loss += loss.item()
        # Calculate AUC and precision
        val_loss = val_loss / val_loader.batch_count()
        val_auc = roc_auc_score(val_targets, val_preds)
        val_precision = average_precision_score(val_targets, val_preds)
        return val_loss, val_auc, val_precision

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Params:
            - data: {   "alpha_chains": alpha chains in a batch,
                        "beta_chains": beta chains in a batch,
                        "peptides": peptides in a batch,
                        "categorical_features": categorical features,
                        "alpha_chain_lens": actual aas sequence length of alpha chains,
                        "beta_chain_lens": actual aas sequence length of beta chains,
                        "peptide_lens": actual aas sequence length of peptides,
                        "labels" : The label of each TCR-Peptide pair
                        }

        Returns:
            - prediction : Torch.tensor with shape (batch_size,1),
                    Model's predictions
            - interaction_map: Torch.tensor with shape (batch_size, tcr_len, peptide_len),
                    Generated interaction maps
            - target : Torch.tensor with shape (batch_size,1),
                    True labels
            - cls_loss : Torch.tensor,
                    Classification loss
            - L1_loss : Torch.tensor,
                    L1 regularization of the interaction map
        """
        self.optimizer.zero_grad()  # Initialize the gradient to zero

        prediction, alpha_interaction_map, beta_interaction_map, _ = self.model(data)
        target = data["labels"]
        # Calculate classification loss with sample weight.
        cls_loss = self.criterion(prediction.squeeze(), target.squeeze())
        cls_loss_with_weight = cls_loss * data["sample_weight"]
        cls_loss_with_weight = cls_loss_with_weight.mean()
        # Add L1 regularization of the interaction map to the final loss.
        L1_loss = self.L1_weight * (
            torch.sum(torch.abs(alpha_interaction_map))
            / (len(torch.nonzero(data["alpha_chain_lens"])) + 1)
            + torch.sum(torch.abs(beta_interaction_map))
            / (len(torch.nonzero(data["beta_chain_lens"])) + 1)
        )
        loss = cls_loss_with_weight + L1_loss
        loss.backward()  # Backwarding loss
        self.optimizer.step()  # Update network weight
        self.model.clip()
        return prediction, target, cls_loss.mean(), cls_loss_with_weight, L1_loss


class LossesLogger(object):
    """
    Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path):
            os.remove(file_path)

        self.logger = logging.getLogger(file_path)
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(
            [
                "Epoch",
                "Train_total_loss",
                "Train_cls_loss",
                "Train_cls_loss_with_weight",
                "Train_L1_loss",
                "Train_auc",
                "Train_ap",
                "Val_loss",
                "Val_auc",
                "Val_ap",
                "Test_loss",
                "Test_auc",
                "Test_ap",
            ]
        )
        self.logger.debug(header)

    def log(self, *args):
        """Write to the log file """

        log_string = ",".join([str(value) for value in args])
        self.logger.debug(log_string)

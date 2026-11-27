# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from .SMITED_utils import (
    CustomDataset,
    CustomDatasetMultitask,
    RMSELoss,
    normalize_smiles,
)

import pandas as pd
import numpy as np

import random
import os
import logging
import shutil
from tqdm import tqdm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from scipy import stats
from .SMITED_utils import RMSE, sensitivity, specificity


class Trainer:
    def __init__(
        self,
        raw_data,
        dataset_name,
        target,
        batch_size,
        hparams,
        target_metric="rmse",
        seed=0,
        smi_ted_version=None,
        checkpoints_folder="./checkpoints",
        results_folder="./results",
        restart_filename=None,
        save_every_epoch=False,
        save_ckpt=True,
        device="cpu",
        console_logger=None,
    ):
        self.df_train = raw_data[0]
        self.df_valid = raw_data[1]
        self.df_test = raw_data[2]
        self.dataset_name = dataset_name
        self.target = target
        self.batch_size = batch_size
        self.hparams = hparams
        self._prepare_data()

        self.target_metric = target_metric
        self.seed = seed
        self.smi_ted_version = smi_ted_version
        self.checkpoints_folder = checkpoints_folder
        self.results_folder = results_folder
        self.restart_filename = restart_filename
        self.start_epoch = 1
        self.save_every_epoch = save_every_epoch
        self.save_ckpt = save_ckpt
        self.device = device
        self.best_vloss = float("inf")
        self.last_filename = None
        self.console_logger = console_logger if console_logger else logging.getLogger()
        self._set_seed(seed)

        self.history = {"train_loss": [], "val_loss": []}

    def _prepare_data(self):
        self.df_train["canon_smiles"] = self.df_train["smiles"].apply(normalize_smiles)
        self.df_valid["canon_smiles"] = self.df_valid["smiles"].apply(normalize_smiles)
        self.df_test["canon_smiles"] = self.df_test["smiles"].apply(normalize_smiles)

        self.df_train = self.df_train.dropna(subset=["canon_smiles"])
        self.df_valid = self.df_valid.dropna(subset=["canon_smiles"])
        self.df_test = self.df_test.dropna(subset=["canon_smiles"])

        self.train_loader = DataLoader(
            CustomDataset(self.df_train, self.target),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            CustomDataset(self.df_valid, self.target),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            CustomDataset(self.df_test, self.target),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def compile(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._print_configuration()
        if self.restart_filename:
            self._load_checkpoint(self.restart_filename)
            logging.info("Checkpoint restored!")

    def fit(self, max_epochs=500):
        for epoch in range(self.start_epoch, max_epochs + 1):
            self.console_logger.info(f"\n=====Epoch [{epoch}/{max_epochs}]=====")

            self.model.to(self.device)
            self.model.train()
            train_loss = self._train_one_epoch()

            self.model.eval()
            val_preds, val_loss, val_metrics = self._validate_one_epoch(
                self.valid_loader
            )
            for m in val_metrics.keys():
                self.console_logger.info(
                    f"[VALID] Evaluation {m.upper()}: {round(val_metrics[m], 4)}"
                )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if (
                (val_loss < self.best_vloss) or self.save_every_epoch
            ) and self.save_ckpt:
                if (self.last_filename != None) and (not self.save_every_epoch):
                    os.remove(os.path.join(self.checkpoints_folder, self.last_filename))

                model_name = f"{str(self.model)}-Finetune"
                self.last_filename = f"{model_name}_seed{self.seed}_{self.dataset_name}_epoch={epoch}_valloss={round(val_loss, 4)}.pt"

                self.best_vloss = val_loss

                self.console_logger.info("Saving checkpoint...")
                self._save_checkpoint(epoch, self.last_filename)

    def evaluate(self, verbose=True):
        if verbose:
            logging.info("\n=====Test Evaluation=====")

        from .SMITED_load import load_smi_ted

        vocab_file = "bert_vocab_curated.txt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.dirname(script_dir)

        model_inf = load_smi_ted(
            folder=self.checkpoints_folder,
            ckpt_filename=self.last_filename,
            vocab_filename=vocab_file,
            eval=True,
            vocab_path=vocab_path,
        ).to(self.device)

        model_inf.eval()

        tst_preds, tst_loss, tst_metrics = self._validate_one_epoch(
            self.test_loader, model_inf
        )

        if verbose:
            for m in tst_metrics.keys():
                logging.info(
                    f"[TEST] Evaluation {m.upper()}: {round(tst_metrics[m], 4)}"
                )

    def _train_one_epoch(self):
        raise NotImplementedError

    def _validate_one_epoch(self, data_loader, model=None):
        raise NotImplementedError

    def _print_configuration(self):
        logging.info("----Finetune information----")
        logging.info(f"Dataset:\t {self.dataset_name}")
        logging.info(f"Target:\t\t {self.target}")
        logging.info(f"Batch size:\t {self.batch_size}")
        logging.info(f"LR:\t\t {self._get_lr()}")
        logging.info(f"Device:\t\t {self.device}")
        logging.info(f"Optimizer:\t {self.optimizer.__class__.__name__}")
        logging.info(f"Loss function:\t {self.loss_fn.__class__.__name__}")
        logging.info(f"Seed:\t\t {self.seed}")
        logging.info(f"Train size:\t {self.df_train.shape[0]}")
        logging.info(f"Valid size:\t {self.df_valid.shape[0]}")
        logging.info(f"Test size:\t {self.df_test.shape[0]}")

    def _load_checkpoint(self, filename):
        ckpt_path = os.path.join(self.checkpoints_folder, filename)
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt_dict["MODEL_STATE"])
        self.start_epoch = ckpt_dict["EPOCHS_RUN"] + 1
        self.best_vloss = ckpt_dict["finetune_info"]["best_vloss"]

    def _save_checkpoint(self, current_epoch, filename):
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)

        ckpt_dict = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": current_epoch,
            "hparams": vars(self.hparams),
            "finetune_info": {
                "dataset": self.dataset_name,
                "target`": self.target,
                "batch_size": self.batch_size,
                "lr": self._get_lr(),
                "device": self.device,
                "optim": self.optimizer.__class__.__name__,
                "loss_fn": self.loss_fn.__class__.__name__,
                "train_size": self.df_train.shape[0],
                "valid_size": self.df_valid.shape[0],
                "test_size": self.df_test.shape[0],
                "best_vloss": self.best_vloss,
            },
            "seed": self.seed,
        }

        assert list(ckpt_dict.keys()) == [
            "MODEL_STATE",
            "EPOCHS_RUN",
            "hparams",
            "finetune_info",
            "seed",
        ]

        torch.save(ckpt_dict, os.path.join(self.checkpoints_folder, filename))

    def _set_seed(self, value):
        random.seed(value)
        torch.manual_seed(value)
        np.random.seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(value)
            torch.cuda.manual_seed_all(value)
            cudnn.deterministic = True
            cudnn.benchmark = False

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]


class TrainerRegressor(Trainer):
    def __init__(
        self,
        raw_data,
        dataset_name,
        target,
        batch_size,
        hparams,
        target_metric="rmse",
        seed=0,
        smi_ted_version=None,
        checkpoints_folder="./checkpoints",
        results_folder="./results",
        restart_filename=None,
        save_every_epoch=False,
        save_ckpt=True,
        device="cpu",
        console_logger=None,
    ):
        super().__init__(
            raw_data,
            dataset_name,
            target,
            batch_size,
            hparams,
            target_metric,
            seed,
            smi_ted_version,
            checkpoints_folder,
            results_folder,
            restart_filename,
            save_every_epoch,
            save_ckpt,
            device,
            console_logger,
        )

    def _train_one_epoch(self):
        running_loss = 0.0

        for idx, data in enumerate(pbar := tqdm(self.train_loader)):
            smiles, targets = data
            targets = targets.clone().detach().to(self.device)

            self.optimizer.zero_grad()

            embeddings = self.model.extract_embeddings(smiles).to(self.device)
            outputs = self.model.net(embeddings).squeeze()

            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            pbar.set_description("[TRAINING]")
            pbar.set_postfix(loss=running_loss / (idx + 1))
            pbar.refresh()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader, model=None):
        data_targets = []
        data_preds = []
        running_loss = 0.0

        model = self.model if model is None else model

        with torch.no_grad():
            for idx, data in enumerate(pbar := tqdm(data_loader)):
                smiles, targets = data
                targets = targets.clone().detach().to(self.device)

                embeddings = model.extract_embeddings(smiles).to(self.device)
                predictions = model.net(embeddings).squeeze()

                loss = self.loss_fn(predictions, targets)

                data_targets.append(targets.view(-1))
                data_preds.append(predictions.view(-1))

                running_loss += loss.item()

                pbar.set_description("[EVALUATION]")
                pbar.set_postfix(loss=running_loss / (idx + 1))
                pbar.refresh()

        preds = torch.cat(data_preds, dim=0).cpu().numpy()
        tgts = torch.cat(data_targets, dim=0).cpu().numpy()

        mae = mean_absolute_error(tgts, preds)
        r2 = r2_score(tgts, preds)
        rmse = RMSE(preds, tgts)
        mse = mean_squared_error(tgts, preds)
        spearman = stats.spearmanr(tgts, preds).statistic
        pearson_r, _ = stats.pearsonr(tgts, preds)

        metrics = {
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "mse": mse,
            "spearman": spearman,
            "pearson_r": pearson_r,
        }

        return preds, running_loss / len(data_loader), metrics


class TrainerClassifier(Trainer):
    def __init__(
        self,
        raw_data,
        dataset_name,
        target,
        batch_size,
        hparams,
        target_metric="roc-auc",
        seed=0,
        smi_ted_version=None,
        checkpoints_folder="./checkpoints",
        results_folder="./results",
        restart_filename=None,
        save_every_epoch=False,
        save_ckpt=True,
        device="cpu",
        console_logger=None,
    ):
        super().__init__(
            raw_data,
            dataset_name,
            target,
            batch_size,
            hparams,
            target_metric,
            seed,
            smi_ted_version,
            checkpoints_folder,
            results_folder,
            restart_filename,
            save_every_epoch,
            save_ckpt,
            device,
            console_logger,
        )

    def _train_one_epoch(self):
        running_loss = 0.0

        for idx, data in enumerate(pbar := tqdm(self.train_loader)):
            smiles, targets = data
            targets = targets.clone().detach().to(self.device)

            self.optimizer.zero_grad()

            embeddings = self.model.extract_embeddings(smiles).to(self.device)
            outputs = self.model.net(embeddings).squeeze()

            loss = self.loss_fn(outputs, targets.long())
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            pbar.set_description("[TRAINING]")
            pbar.set_postfix(loss=running_loss / (idx + 1))
            pbar.refresh()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader, model=None):
        data_targets = []
        data_preds = []
        running_loss = 0.0

        model = self.model if model is None else model

        with torch.no_grad():
            for idx, data in enumerate(pbar := tqdm(data_loader)):
                smiles, targets = data
                targets = targets.clone().detach().to(self.device)

                embeddings = model.extract_embeddings(smiles).to(self.device)
                predictions = model.net(embeddings).squeeze()

                loss = self.loss_fn(predictions, targets.long())

                data_targets.append(targets.view(-1))
                data_preds.append(predictions)

                running_loss += loss.item()

                pbar.set_description("[EVALUATION]")
                pbar.set_postfix(loss=running_loss / (idx + 1))
                pbar.refresh()

        preds = torch.cat(data_preds, dim=0).cpu().numpy()
        tgts = torch.cat(data_targets, dim=0).cpu().numpy()

        preds_cpu = F.softmax(torch.tensor(preds), dim=1).cpu().numpy()[:, 1]

        y_pred = np.where(preds_cpu >= 0.5, 1, 0)
        accuracy = accuracy_score(tgts, y_pred)

        sn = sensitivity(tgts, y_pred)

        sp = specificity(tgts, y_pred)

        fpr, tpr, _ = roc_curve(tgts, preds_cpu)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(tgts, preds_cpu)
        prc_auc = auc(recall, precision)

        metrics = {
            "acc": accuracy,
            "roc-auc": roc_auc,
            "prc-auc": prc_auc,
            "sensitivity": sn,
            "specificity": sp,
        }

        return preds, running_loss / len(data_loader), metrics


class TrainerClassifierMultitask(Trainer):
    def __init__(
        self,
        raw_data,
        dataset_name,
        target,
        batch_size,
        hparams,
        target_metric="roc-auc",
        seed=0,
        smi_ted_version=None,
        checkpoints_folder="./checkpoints",
        results_folder="./results",
        restart_filename=None,
        save_every_epoch=False,
        save_ckpt=True,
        device="cpu",
        console_logger=None,
    ):
        super().__init__(
            raw_data,
            dataset_name,
            target,
            batch_size,
            hparams,
            target_metric,
            seed,
            smi_ted_version,
            checkpoints_folder,
            results_folder,
            restart_filename,
            save_every_epoch,
            save_ckpt,
            device,
            console_logger,
        )

    def _prepare_data(self):
        self.df_train["canon_smiles"] = self.df_train["smiles"].apply(normalize_smiles)
        self.df_valid["canon_smiles"] = self.df_valid["smiles"].apply(normalize_smiles)
        self.df_test["canon_smiles"] = self.df_test["smiles"].apply(normalize_smiles)

        self.df_train = self.df_train.dropna(subset=["canon_smiles"])
        self.df_valid = self.df_valid.dropna(subset=["canon_smiles"])
        self.df_test = self.df_test.dropna(subset=["canon_smiles"])

        self.train_loader = DataLoader(
            CustomDatasetMultitask(self.df_train, self.target),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            CustomDatasetMultitask(self.df_valid, self.target),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            CustomDatasetMultitask(self.df_test, self.target),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def _train_one_epoch(self):
        running_loss = 0.0

        for idx, data in enumerate(pbar := tqdm(self.train_loader)):
            smiles, targets, target_masks = data
            targets = targets.clone().detach().to(self.device)

            self.optimizer.zero_grad()

            embeddings = self.model.extract_embeddings(smiles).to(self.device)
            outputs = self.model.net(embeddings, multitask=True).squeeze()
            outputs = outputs * target_masks.to(self.device)

            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            pbar.set_description("[TRAINING]")
            pbar.set_postfix(loss=running_loss / (idx + 1))
            pbar.refresh()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader, model=None):
        data_targets = []
        data_preds = []
        data_masks = []
        running_loss = 0.0

        model = self.model if model is None else model

        with torch.no_grad():
            for idx, data in enumerate(pbar := tqdm(data_loader)):
                smiles, targets, target_masks = data
                targets = targets.clone().detach().to(self.device)

                embeddings = model.extract_embeddings(smiles).to(self.device)
                predictions = model.net(embeddings, multitask=True).squeeze()
                predictions = predictions * target_masks.to(self.device)

                loss = self.loss_fn(predictions, targets)

                data_targets.append(targets)
                data_preds.append(predictions)
                data_masks.append(target_masks)

                running_loss += loss.item()

                pbar.set_description("[EVALUATION]")
                pbar.set_postfix(loss=running_loss / (idx + 1))
                pbar.refresh()

        preds = torch.cat(data_preds, dim=0)
        tgts = torch.cat(data_targets, dim=0)
        mask = torch.cat(data_masks, dim=0)
        mask = mask > 0

        roc_aucs = []
        prc_aucs = []
        sns = []
        sps = []
        num_tasks = len(self.target)
        for idx in range(num_tasks):
            actuals_task = torch.masked_select(
                tgts[:, idx], mask[:, idx].to(self.device)
            )
            preds_task = torch.masked_select(
                preds[:, idx], mask[:, idx].to(self.device)
            )

            y_pred = np.where(preds_task.cpu().detach() >= 0.5, 1, 0)
            accuracy = accuracy_score(actuals_task.cpu().numpy(), y_pred)

            sn = sensitivity(actuals_task.cpu().numpy(), y_pred)

            sp = specificity(actuals_task.cpu().numpy(), y_pred)

            roc_auc = roc_auc_score(
                actuals_task.cpu().numpy(), preds_task.cpu().numpy()
            )

            precision, recall, thresholds = precision_recall_curve(
                actuals_task.cpu().numpy(), preds_task.cpu().numpy()
            )
            prc_auc = auc(recall, precision)

            sns.append(sn)
            sps.append(sp)
            roc_aucs.append(roc_auc)
            prc_aucs.append(prc_auc)
        average_sn = torch.mean(torch.tensor(sns))
        average_sp = torch.mean(torch.tensor(sps))
        average_roc_auc = torch.mean(torch.tensor(roc_aucs))
        average_prc_auc = torch.mean(torch.tensor(prc_aucs))

        metrics = {
            "acc": accuracy,
            "roc-auc": average_roc_auc.item(),
            "prc-auc": average_prc_auc.item(),
            "sensitivity": average_sn.item(),
            "specificity": average_sp.item(),
        }

        return preds.cpu().numpy(), running_loss / len(data_loader), metrics

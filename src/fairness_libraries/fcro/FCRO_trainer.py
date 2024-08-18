from typing import Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation_metrics import EvaluationMetrics
from src.fairness_libraries.fcro.FCRO_loss import ColOrthLoss, RowOrthLoss, check_utility
from src.fairness_libraries.fcro.FCRO_constants import LOSS_COL_WEIGHT, LOSS_ROW_WEIGHT, FCRO_SUBSPACE_THRESHOLD

from src.model_training import ModelTrainer


class FCROTrainer(ModelTrainer):

    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, device, model_path,
                 num_classes_target,
                 num_classes_protected_feature,
                 model_t,
                 model_a,
                 mode,
                 batch_size):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target, device,
                         model_path)
        self.mode = mode
        self.logger = logger
        self.model_t = model_t.to(device)
        self.model_a = model_a.to(device)
        self.batch_size = batch_size
        self.num_classes_protected_feature = num_classes_protected_feature

        self.multiclass_metrics = False
        if self.mode == 0 and self.num_classes_protected_feature > 2:
            self.multiclass_metrics = True
        elif self.mode == 1 and self.num_classes_target > 2:
            self.multiclass_metrics = True

        self.lr = 0.001

        self.moving_base = False

        self.criterion = nn.CrossEntropyLoss()

        if num_classes_target == 2:
            self.conditional = True
        else:
            self.conditional = False

        if self.mode == 1:
            for param in self.model_a.parameters():
                param.requires_grad = False

            self.row_criterion = RowOrthLoss(
                conditional=self.conditional,
                margin=0,
            ).to(self.device)


            if not self.moving_base:
                U = self.generate_sensitive_subspace(
                    DataLoader(
                        self.train_dataloader.dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        drop_last=False
                    )
                )
            else:
                U = None

            self.col_criterion = ColOrthLoss(
                U,
                conditional=self.conditional,
                margin=0,
                moving_base=self.moving_base,
            ).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.model_t.parameters(), lr=self.lr
            )

        elif self.mode == 0:
            self.optimizer = torch.optim.Adam(
                self.model_a.parameters(),
                lr=self.lr)
        elif self.mode == -1:
            pass
        else:
            raise NotImplementedError

    def train_step(self, images, targets, protected_features, epoch):
        x, y, a = images.to(self.device), targets.to(self.device), protected_features.to(self.device)
        y, a = y.long(), a.long()
        #loss_dict = {}
        loss = 0.0

        if self.mode == 0:
            self.model_a.train()

            out, emb = self.model_a(x)

            loss = self.criterion(out, a)
            #loss_dict[f"loss_sa"] = loss.item()

        elif self.mode == 1:
            self.model_a.eval()
            self.model_t.train()

            out, emb = self.model_t(x)
            with torch.no_grad():
                out_a, emb_a = self.model_a(x)

            loss_sup = self.criterion(out, y)
            #loss_dict["loss_sup"] = loss_sup.item()
            loss = loss + loss_sup

            if LOSS_COL_WEIGHT:
                if self.moving_base:
                    loss_col = (
                        self.col_criterion(emb, y, emb_a, epoch) * LOSS_COL_WEIGHT
                    )
                else:
                    loss_col = self.col_criterion(emb, y) * LOSS_COL_WEIGHT
                loss = loss + loss_col
                #loss_dict["loss_col"] = loss_col.item()

            if LOSS_ROW_WEIGHT:
                loss_row = self.row_criterion(emb, emb_a.detach(), y) * LOSS_ROW_WEIGHT
                loss = loss + loss_row
                #loss_dict["loss_row"] = loss_row.item()

        #loss_dict["loss"] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return out, loss

    def train(self):

        best_valid_f1 = 0
        best_valid_eod = float("inf")
        best_loss = float("inf")

        train_y_true = []
        train_y_pred = []
        train_y_prob = []
        train_y_protected_feature = []

        self.logger.info("Training started.")
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            self.model.train()
            for images, targets, protected_features in tqdm(self.train_dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)

                self.logger.debug(f"------------ EPOCH {epoch} ------------")
                self.logger.debug(f"IMAGES: \n {images}")
                self.logger.debug(f"TARGETS: \n {targets}")
                self.logger.debug(f"PROTECTED FEATURES: \n {protected_features}")
                self.logger.debug(f"SHAPES (images, targets): {images.size()}, {targets.size()}")

                outputs, train_loss = self.train_step(images, targets, protected_features, epoch)

                probabilities = F.softmax(outputs, dim=1)
                train_y_prob.extend(probabilities.detach().cpu().numpy())
                _, y_pred = outputs.max(dim=1)
                train_y_pred.extend(y_pred.detach().cpu().numpy())
                if self.mode == 0:
                    train_y_true.extend(protected_features.detach().cpu().numpy())
                    train_y_protected_feature.extend(targets.detach().cpu().numpy())
                elif self.mode == 1:
                    train_y_true.extend(targets.detach().cpu().numpy())
                    train_y_protected_feature.extend(protected_features.detach().cpu().numpy())

                self.logger.debug(f"OUTPUTS: \n {outputs}")
                self.logger.debug(f"SHAPES outputs: {outputs.size()}")
                self.logger.debug(f"LOSS: {train_loss}")

                running_loss += train_loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_dataloader.dataset)

            # Log train process
            metrics = EvaluationMetrics(train_y_true, train_y_pred, train_y_prob, train_y_protected_feature, logger=self.logger)


            if self.multiclass_metrics:
                eod_max, eod_mean = metrics.get_equalized_odds_difference()
                dp_max, dp_mean = metrics.get_demographic_parity_difference()
                train_progress_metrics = {
                    f"train_acc": metrics.get_accuracy(),
                    f"train_f1": metrics.get_f1(),
                    f"train_eod_max": eod_max,
                    f"train_eod_mean": eod_mean,
                    f"train_eod_strict": metrics.get_strict_multiclass_equalized_odds(),
                    f"train_dp_max": dp_max,
                    f"train_dp_mean": dp_mean,
                    "train_loss": epoch_loss
                }
                self.writer.add_scalar('EOD_max/train', train_progress_metrics["train_eod_max"], epoch)
                self.writer.add_scalar('EOD_mean/train', train_progress_metrics["train_eod_mean"], epoch)
                self.writer.add_scalar('EOD_strict/train', train_progress_metrics["train_eod_strict"], epoch)
                self.writer.add_scalar('DP_max/train', train_progress_metrics["train_dp_max"], epoch)
                self.writer.add_scalar('DP_mean/train', train_progress_metrics["train_dp_mean"], epoch)
            else:
                train_progress_metrics = {
                    f"train_acc": metrics.get_accuracy(),
                    f"train_f1": metrics.get_f1(),
                    f"train_eod": metrics.get_equalized_odds_difference(),
                    f"train_dp": metrics.get_demographic_parity_difference(),
                    "train_loss": epoch_loss
                }
                self.writer.add_scalar('EOD/train', train_progress_metrics["train_eod"], epoch)
                self.writer.add_scalar('DP/train', train_progress_metrics["train_dp"], epoch)

            self.train_process.append(train_progress_metrics)

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_progress_metrics["train_acc"], epoch)
            self.writer.add_scalar('F1/train', train_progress_metrics["train_f1"], epoch)

            # Log validation process
            valid_metrics = self.validate_model(self.valid_dataloader, "valid")
            self.valid_process.append(valid_metrics)
            valid_acc = valid_metrics["valid_acc"]
            valid_f1 = valid_metrics["valid_f1"]

            # Log to tensorboard
            self.writer.add_scalar('Accuracy/validation', valid_acc, epoch)
            self.writer.add_scalar('F1/validation', valid_f1, epoch)

            if self.multiclass_metrics:
                valid_eod = valid_metrics["valid_eod_max"]
                valid_dp = valid_metrics["valid_dp_max"]
                self.writer.add_scalar('EOD_max/validation', valid_metrics["valid_eod_max"], epoch)
                self.writer.add_scalar('EOD_mean/validation', valid_metrics["valid_eod_mean"], epoch)
                self.writer.add_scalar('EOD_strict/validation', valid_metrics["valid_eod_strict"], epoch)
                self.writer.add_scalar('DP_max/validation', valid_metrics["valid_dp_max"], epoch)
                self.writer.add_scalar('DP_mean/validation', valid_metrics["valid_dp_mean"], epoch)
            else:
                valid_eod = valid_metrics["valid_eod"]
                valid_dp = valid_metrics["valid_dp"]
                self.writer.add_scalar('EOD/validation', valid_eod, epoch)
                self.writer.add_scalar('DP/validation', valid_dp, epoch)

            self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}, Valid DP: {valid_dp:.4f}, Valid EOD: {valid_eod:.4f}")

            # Save best f1 model
            if best_valid_f1 < valid_f1 and (valid_f1 != 0 or epoch >= self.num_epochs*0.2):
                best_valid_f1 = valid_f1
                self.save_model("best_f1", epoch)

            # Save best eod model
            if best_valid_eod > valid_eod and (valid_eod != 0 or epoch >= self.num_epochs*0.2):
                best_valid_eod = valid_eod
                self.save_model("best_eod", epoch)

            # Save best loss model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                self.save_model("best_loss", epoch)

            self.current_epoch += 1

        self.save_model_training()
        self.logger.info("Training finished.")
        pass

    def validate_model(self, dataloader, mode: Literal['train', 'valid', 'test']):
        self.model.eval()
        y_true, y_pred, y_prob, y_biases = self._compute_predictions(dataloader)
        metrics = EvaluationMetrics(y_true, y_pred, y_prob, y_biases, logger=self.logger)
        TPR, TNR, FPR, FNR = metrics.get_prediction_rates()
        if self.multiclass_metrics:
            eod_max, eod_mean = metrics.get_equalized_odds_difference()
            dp_max, dp_mean = metrics.get_demographic_parity_difference()
            progress_metrics = {
                f"{mode}_acc": metrics.get_accuracy(),
                f"{mode}_f1": metrics.get_f1(),
                f"{mode}_precision": metrics.get_precision(),
                f"{mode}_auc": metrics.get_roc_auc_score(),
                f"{mode}_eod_max": eod_max,
                f"{mode}_eod_mean": eod_mean,
                f"{mode}_eod_strict": metrics.get_strict_multiclass_equalized_odds(),
                f"{mode}_dp_max": dp_max,
                f"{mode}_dp_mean": dp_mean,
                f"{mode}_TPR": TPR,
                f"{mode}_TNR": TNR,
                f"{mode}_FPR": FPR,
                f"{mode}_FNR": FNR,
                f"{mode}_MCC": metrics.get_mcc()
            }
        else:
            progress_metrics = {
                f"{mode}_acc": metrics.get_accuracy(),
                f"{mode}_f1": metrics.get_f1(),
                f"{mode}_precision": metrics.get_precision(),
                f"{mode}_auc": metrics.get_roc_auc_score(),
                f"{mode}_eod": metrics.get_equalized_odds_difference(),
                f"{mode}_dp": metrics.get_demographic_parity_difference(),
                f"{mode}_TPR": TPR,
                f"{mode}_TNR": TNR,
                f"{mode}_FPR": FPR,
                f"{mode}_FNR": FNR,
                f"{mode}_MCC": metrics.get_mcc()
            }
        return progress_metrics

    def _compute_predictions(self, dataloader: DataLoader) -> Tuple[List, List, List, List]:
        total_labels = []
        total_outputs = []
        total_pred = []
        total_biases = []

        with torch.no_grad():
            for images, targets, protected_features in tqdm(dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)
                if self.mode == 0:
                    outputs = self.validate_sensitive(images)
                    protected = targets
                    total_labels.extend(protected_features.detach().cpu().numpy())
                elif self.mode == 1:
                    outputs = self.validate_target(images)
                    protected = protected_features
                    total_labels.extend(targets.detach().cpu().numpy())
                else:
                    raise NotImplementedError

                probabilities = F.softmax(outputs, dim=1)
                total_outputs.extend(probabilities.detach().cpu().numpy())
                pred = self._compute_prediction_from_output(outputs)
                total_pred.extend(pred.detach().cpu().numpy())
                total_biases.extend(protected.detach().cpu().numpy())

        if self.post_processing_function is not None:
            self.logger.info("Applied post processing")
            total_pred = self.post_processing_function(total_labels, total_pred, total_biases)

        return total_labels, total_pred, total_outputs, total_biases

    def validate_target(self, images):
        self.model_t.eval()
        logits, _ = self.model_t(images)
        return logits


    def validate_sensitive(self, images):
        self.model_a.eval()
        logits, _ = self.model_a(images)
        return logits

    def generate_sensitive_subspace(self, dataloader):
        assert self.mode == 1, "Subspace is needed only when training target head."

        self.logger.info(
            f"Building static subspace for sensitive representations on {len(dataloader.dataset)} samples."
        )

        emb = []
        targets = []
        with torch.no_grad():
            for input, target, _ in dataloader:
                input = input.to(self.device)
                emb.append(self.model_a(input)[1])
                targets.append(target)

        emb = torch.concat(emb, dim=0).cpu()
        targets = torch.concat(targets, dim=0).squeeze().cpu()

        U_list = []
        for i in range(int(self.conditional) + 1):
            if self.conditional:
                indices = torch.where(targets == i)[0]
                emb_sub = torch.index_select(emb, 0, indices)
            else:
                emb_sub = emb

            U, S, _ = torch.linalg.svd(emb_sub.T, full_matrices=False)

            sval_ratio = (S**2) / (S**2).sum()
            r = (torch.cumsum(sval_ratio, -1) <= FCRO_SUBSPACE_THRESHOLD).sum()
            U_list.append(U[:, :r])

        return U_list

    def get_models(self):
        return self.model_t, self.model_a
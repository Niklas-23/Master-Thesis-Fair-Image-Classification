import os
from typing import Tuple, List, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from src.constants import BENCHMARK_NUM_EPOCHS, BENCHMARK_NUM_EPOCHS_CELEBA
from src.evaluation_metrics import EvaluationMetrics
import torch.nn.functional as F

class ModelTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target, device="cpu", model_path=None, post_processing_function=None):

        self.base_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"

        if model_path is None:
            raise AssertionError("Model path must be defined")
        else:
            self.model_path = f"{self.base_path}/model_weights/{model_path}"
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            #elif os.listdir(self.model_path):
                #raise AssertionError("Model path must not empty")

        self.device = device
        self.num_classes_target = num_classes_target
        self.logger = logger
        self.post_processing_function = post_processing_function

        # Dataloaders
        self.train_dataloader: DataLoader = train_dataloader
        self.valid_dataloader: DataLoader = valid_dataloader
        self.test_dataloader: DataLoader = test_dataloader

        # Essential model and training choices
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = self._default_loss_function

        # Train configuration
        self.num_epochs = BENCHMARK_NUM_EPOCHS_CELEBA if "celeba" in self.model_path else BENCHMARK_NUM_EPOCHS
        self.current_epoch = 0

        # Track train progress
        self.train_process = []
        self.valid_process = []

        self.writer = SummaryWriter(f"{self.base_path}/tensorboard_runs/{model_path}")

    def _default_loss_function(self, outputs, targets, protected_features):
        return nn.CrossEntropyLoss().forward(outputs, targets)

    def update_data_loaders(self, train_dataloader, valid_dataloader, test_dataloader):
        self.train_dataloader: DataLoader = train_dataloader
        self.valid_dataloader: DataLoader = valid_dataloader
        self.test_dataloader: DataLoader = test_dataloader

    def set_model_path(self, new_model_path):
        self.model_path = f"{self.base_path}/{new_model_path}"
        pass

    def set_optimizer(self, optimizer_type: str, **kwargs):
        optimizer_cls = getattr(torch.optim, optimizer_type)
        self.optimizer = optimizer_cls(self.model.parameters, **kwargs)
        pass

    def set_loss_function(self, new_loss_function):
        self.loss_function = new_loss_function
        pass

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()  # Reset gradients to zero
        outputs = self.model(images)  # Compute model predictions

        train_loss = self.loss_function(outputs, targets.long(), protected_features)
        train_loss.backward()  # Compute gradients
        self.optimizer.step()  # Update model parameters

        return outputs, train_loss

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
                train_y_true.extend(targets.detach().cpu().numpy())
                train_y_protected_feature.extend(protected_features.detach().cpu().numpy())

                self.logger.debug(f"OUTPUTS: \n {outputs}")
                self.logger.debug(f"SHAPES outputs: {outputs.size()}")
                self.logger.debug(f"LOSS: {train_loss}")

                running_loss += train_loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_dataloader.dataset)

            # Log train process
            metrics = EvaluationMetrics(train_y_true, train_y_pred, train_y_prob, train_y_protected_feature, logger=self.logger)

            if self.num_classes_target > 2:
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

            if self.num_classes_target > 2:
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

    def _batch_to_device(self, images, targets, protected_features):
        images, targets, protected_features = images.to(self.device), targets.to(self.device), protected_features.to(
            self.device)
        return images, targets, protected_features

    def save_model(self, model_name: str, epoch):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch
        }
        file_path = self.model_path + f"/{model_name}_state.pth"
        torch.save(state, file_path)
        self.logger.info(f"Saved model {model_name} to {file_path}.")

    def load_model(self, extra_model_path=None, extra_model_version=None):
        if extra_model_path is None and extra_model_version is None:
            if os.path.exists(self.model_path + f"/final_state.pth"):
                state = torch.load(self.model_path + f"/final_state.pth")
                self.logger.info(f"Final state model loading.")
            else:
                state = torch.load(self.model_path + f"/best_loss.pth")
                self.logger.info(f"Best loss model loading.")
        else:
            if extra_model_version is None and extra_model_path is not None:
                state = torch.load(extra_model_path)
            else:
                state = torch.load(self.model_path + f"/{extra_model_version}.pth")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.current_epoch = state["epoch"]
        self.logger.info(f"Model has been loaded from {self.model_path}.")

    def save_model_training(self):
        self.save_model("final", self.num_epochs)
        self.writer.flush()
        self.writer.close()
        df_train = pd.DataFrame(self.train_process)
        df_valid = pd.DataFrame(self.valid_process)
        df = pd.concat([df_train, df_valid], axis=1)
        df.to_csv(self.model_path + "/metrics.csv", index=False)
        self.logger.info("Model and metrics saved.")

    def validate_model(self, dataloader, mode: Literal['train', 'valid', 'test']):
        self.model.eval()
        y_true, y_pred, y_prob, y_biases = self._compute_predictions(dataloader)
        metrics = EvaluationMetrics(y_true, y_pred, y_prob, y_biases, logger=self.logger)
        TPR, TNR, FPR, FNR = metrics.get_prediction_rates()
        if self.num_classes_target > 2:
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

    def test(self):
        test_metrics = self.validate_model(self.test_dataloader, "test")
        self.logger.info(f"Test Accuracy: {test_metrics['test_acc']:.4f}")

    def _compute_predictions(self, dataloader: DataLoader) -> Tuple[List, List, List, List]:
        total_labels = []
        total_outputs = []
        total_pred = []
        total_biases = []

        with torch.no_grad():
            for images, targets, protected_features in tqdm(dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                total_outputs.extend(probabilities.detach().cpu().numpy())
                pred = self._compute_prediction_from_output(outputs)
                total_pred.extend(pred.detach().cpu().numpy())
                total_biases.extend(protected_features.detach().cpu().numpy())
                total_labels.extend(targets.detach().cpu().numpy())

        if self.post_processing_function is not None:
            self.logger.info("Applied post processing")
            total_pred = self.post_processing_function(total_labels, total_pred, total_biases)

        return total_labels, total_pred, total_outputs, total_biases

    def compute_train_eval_predictions(self):
        complete_output_logits = []
        self.model.eval()
        for images, targets, protected_features in tqdm(self.train_dataloader):
            images, targets, protected_features = self._batch_to_device(images, targets, protected_features)
            outputs = self.model(images)
            complete_output_logits.extend(outputs)
        return complete_output_logits

    def _compute_prediction_from_output(self, output):
        _, y_pred = output.max(dim=1)
        return y_pred

    def generate_latent_embedding(self, test_dataloader):
        latent_embeddings = []
        with torch.no_grad():
            for images, targets, protected_features in tqdm(test_dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)
                latent_emb = self.model.get_intermediate_feature_representations(images)
                latent_emb = latent_emb.detach().cpu().numpy()
                latent_emb = np.hstack((latent_emb, targets.detach().cpu().numpy().reshape(-1, 1)))
                latent_emb = np.hstack((latent_emb, protected_features.detach().cpu().numpy().reshape(-1, 1)))
                latent_embeddings.extend(latent_emb)
        return latent_embeddings

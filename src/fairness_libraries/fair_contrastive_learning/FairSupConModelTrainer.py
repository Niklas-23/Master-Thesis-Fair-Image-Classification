import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.fairness_libraries.fair_contrastive_learning.FairSupConModels import FairSupConResNet18
from src.model_training import ModelTrainer
import torch.nn.functional as F

NUM_EPOCHS_CONTRASTIVE_TRAINING = 100
NUM_EPOCHS_CONTRASTIVE_TRAINING_CELEBA = 50


class FairContrastiveModelTrainer(ModelTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                 device="cpu", model_path=None):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                         device=device, model_path=model_path)
        self.num_epochs = NUM_EPOCHS_CONTRASTIVE_TRAINING_CELEBA if "celeba" in self.model_path else NUM_EPOCHS_CONTRASTIVE_TRAINING
    def _batch_to_device(self, images, targets, protected_features):
        images = torch.cat([images[0], images[1]], dim=0)
        images, targets, protected_features = images.to(self.device), targets.to(self.device), protected_features.to(
            self.device)
        return images, targets, protected_features

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()
        outputs = self.model(images)

        batch_size = targets.shape[0]
        f1, f2 = torch.split(outputs, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        train_loss = self.loss_function(features, targets.float(), protected_features)

        train_loss.backward()
        self.optimizer.step()

        return outputs, train_loss

    def train(self):

        self.logger.info("Contrastive Training started.")
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

                self.logger.debug(f"OUTPUTS: \n {outputs}")
                self.logger.debug(f"SHAPES outputs: {outputs.size()}")
                self.logger.debug(f"LOSS: {train_loss}")

                running_loss += train_loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_dataloader.dataset)

            # Log train process
            train_progress = {"train_loss": epoch_loss}
            self.train_process.append(train_progress)

            self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

            self.current_epoch += 1

        self.save_model_training()
        self.logger.info("Contrastive Training finished.")
        pass

    def save_model(self, model_name: str, epoch):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        file_path = self.model_path + f"/{model_name}_state.pth"
        torch.save(state, file_path)
        self.logger.info(f"Saved model {model_name} to {file_path}.")

    def save_model_training(self):
        self.writer.flush()
        self.writer.close()
        self.save_model("final_contrastive_model", self.current_epoch)
        df_train = pd.DataFrame(self.train_process)
        df_train.to_csv(self.model_path + "/loss.csv", index=False)
        self.logger.info("Contrastive Model and loss saved.")


class FairContrastiveClassifierModelTrainer(ModelTrainer):
    def __init__(self, classifier_model, contrastive_model: FairSupConResNet18, train_dataloader, valid_dataloader, test_dataloader, logger,
                 num_classes_target,
                 device="cpu", model_path=None):
        super().__init__(classifier_model, train_dataloader, valid_dataloader, test_dataloader, logger,
                         num_classes_target,
                         device=device, model_path=model_path)
        self.contrastive_model = contrastive_model

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()
        with torch.no_grad():
            features = self.contrastive_model.encoder(images)
        outputs = self.model(features.detach())
        train_loss = self.loss_function(outputs, targets.long(), protected_features)
        train_loss.backward()
        self.optimizer.step()

        return outputs, train_loss

    def _compute_predictions(self, dataloader: DataLoader) -> Tuple[List, List, List, List]:
        total_labels = []
        total_outputs = []
        total_pred = []
        total_biases = []

        with torch.no_grad():
            for images, targets, protected_features in tqdm(dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)
                with torch.no_grad():
                    features = self.contrastive_model.encoder(images)
                outputs = self.model(features.detach())
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
            features = self.encoder(images)
            outputs = self.model(features)
            complete_output_logits.extend(outputs)
        return complete_output_logits

    def generate_latent_embedding(self, test_dataloader):
        latent_embeddings = []
        with torch.no_grad():
            for images, targets, protected_features in tqdm(test_dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)
                latent_emb = self.contrastive_model.encoder(images)
                latent_emb = latent_emb.detach().cpu().numpy()
                latent_emb = np.hstack((latent_emb, targets.detach().cpu().numpy().reshape(-1, 1)))
                latent_emb = np.hstack((latent_emb, protected_features.detach().cpu().numpy().reshape(-1, 1)))
                latent_embeddings.extend(latent_emb)
        return latent_embeddings

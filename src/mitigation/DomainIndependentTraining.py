from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.benchmark_models.ResNet18 import ResNet18
from src.datasets.dataset_wrapper import DatasetWrapper
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer
import torch.nn.functional as F


class DomainIndependentTrainingBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)

    def apply_pre_processing(self):
        self.logger.info("Start domain independent training benchmark.")
        pass

    def train_model(self):
        model_trainer = DomainIndependentModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.logger, self.num_classes_target, self.num_classes_protected_feature, device=self.device, model_path=self.model_path)
        model_trainer.set_loss_function(DomainIndependentLoss(num_classes_target=self.num_classes_target, num_classes_protected_feature=self.num_classes_protected_feature))
        model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        domain_independent_model_trainer = DomainIndependentModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                      self.test_dataloader,
                                                      self.test_logger, self.num_classes_target,
                                                      self.num_classes_protected_feature, device=self.device,
                                                      model_path=self.model_path)

        domain_independent_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = domain_independent_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Domain Independent: {test_results}")
        return "Domain Independent", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        domain_independent_model_trainer = DomainIndependentModelTrainer(self.model, self.train_dataloader,
                                                                         self.valid_dataloader,
                                                                         self.test_dataloader,
                                                                         self.test_logger, self.num_classes_target,
                                                                         self.num_classes_protected_feature,
                                                                         device=self.device,
                                                                         model_path=self.model_path)

        domain_independent_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = domain_independent_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Domain Independent", latent_embedding


class DomainIndependentModelTrainer(ModelTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                 num_classes_protected_feature, device, model_path):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target, device,
                         model_path)
        self.num_classes_protected_feature = num_classes_protected_feature

    def _compute_predictions(self, dataloader: DataLoader) -> Tuple[List, List, List, List]:
        total_labels = []
        total_y_pred = []
        total_biases = []
        total_outputs = []

        with torch.no_grad():
            for images, labels, biases in tqdm(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                y_pred = self._compute_prediction_from_output(output)
                output_sum = torch.zeros(output.size(0), self.num_classes_target, device=output.device)
                for i in range(self.num_classes_protected_feature):
                    output_sum += output[:, self.num_classes_target * i: self.num_classes_target * (i + 1)]
                probabilities = F.softmax(output_sum, dim=1)
                total_outputs.extend(probabilities.detach().cpu().numpy())
                total_y_pred.extend(y_pred.detach().cpu().numpy())
                total_biases.extend(biases.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())

        return total_labels, total_y_pred, total_outputs, total_biases

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()
        outputs = self.model(images)

        train_loss = self.loss_function(outputs, targets.long(), protected_features)
        train_loss.backward()
        self.optimizer.step()

        output_sum = torch.zeros(outputs.size(0), self.num_classes_target, device=outputs.device)
        for i in range(self.num_classes_protected_feature):
            output_sum += outputs[:, self.num_classes_target * i: self.num_classes_target * (i + 1)]

        return output_sum, train_loss

    def compute_train_eval_predictions(self):
        complete_output_logits = []
        self.model.eval()
        for images, labels, biases in tqdm(self.train_dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            output_sum = torch.zeros(output.size(0), self.num_classes_target, device=output.device)
            for i in range(self.num_classes_protected_feature):
                output_sum += output[:, self.num_classes_target * i: self.num_classes_target * (i + 1)]
            complete_output_logits.extend(output_sum)
        return complete_output_logits

    def _compute_prediction_from_output(self, output):
        output_sum = torch.zeros(output.size(0), self.num_classes_target, device=output.device)
        for i in range(self.num_classes_protected_feature):
            output_sum += output[:, self.num_classes_target * i: self.num_classes_target * (i + 1)]
        _, y_pred = output_sum.max(dim=1)
        return y_pred


class DomainIndependentLoss(nn.Module):
    def __init__(self, num_classes_target=2, num_classes_protected_feature=2):
        super().__init__()
        self.num_classes_target = num_classes_target
        self.num_classes_protected_feature = num_classes_protected_feature
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, output, target_labels, protected_features):

        output_sum = torch.zeros(output.size(0), self.num_classes_target, device=output.device)

        for i in range(self.num_classes_protected_feature):
            domain_label_hit = protected_features == i
            output_sum += output[:, self.num_classes_target*i: self.num_classes_target*(i+1)]*domain_label_hit.view(-1,1).float()

        loss = self.loss_function(output_sum, target_labels.long())

        return loss

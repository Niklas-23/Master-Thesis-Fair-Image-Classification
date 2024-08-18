import os

import torch
from torch.utils.data import DataLoader

from src.fairness_libraries.fair_contrastive_learning.FairSupConLoss import FairSupConLoss
from src.fairness_libraries.fair_contrastive_learning.FairSupConModelTrainer import FairContrastiveModelTrainer, FairContrastiveClassifierModelTrainer
from src.mitigation.benchmark_interface import BenchmarkInterface


class FairContrastiveLearningBenchmark(BenchmarkInterface):
    def __init__(self, contrastive_model, classifier_model, contrastive_dataset, classifier_dataset, model_path: str, device="cpu", batch_size=32):
        super().__init__(None, classifier_dataset, model_path, device, batch_size)
        self.contrastive_model = contrastive_model
        self.contrastive_dataset = contrastive_dataset
        self.classifier_dataset = classifier_dataset
        self.classifier_model = classifier_model

    def apply_pre_processing(self):
        self.logger.info("Start fair contrastive learning benchmark.")
        pass

    def train_model(self):
        train_dataset, valid_dataset, test_dataset = self.contrastive_dataset.get_datasets()
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)

        contrastive_model_trainer = FairContrastiveModelTrainer(self.contrastive_model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.logger, self.num_classes_target, device=self.device, model_path=self.model_path+"_contrastive")
        contrastive_model_trainer.set_loss_function(FairSupConLoss(device=self.device))
        contrastive_model_trainer.train()

        train_dataset, valid_dataset, test_dataset = self.classifier_dataset.get_datasets()
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        classifier_model_trainer = FairContrastiveClassifierModelTrainer(self.classifier_model, self.contrastive_model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.logger, self.num_classes_target, device=self.device, model_path=self.model_path+"_classifier")
        classifier_model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        self._load_contrastive_model()
        classifier_model_trainer = FairContrastiveClassifierModelTrainer(self.classifier_model, self.contrastive_model,
                                                                         self.train_dataloader, self.valid_dataloader,
                                                                         self.test_dataloader,
                                                                         self.test_logger, self.num_classes_target,
                                                                         device=self.device,
                                                                         model_path=self.model_path + "_classifier")

        classifier_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = classifier_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Contrastive Training: {test_results}")
        return "Contrastive Training", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        self._load_contrastive_model()
        classifier_model_trainer = FairContrastiveClassifierModelTrainer(self.classifier_model, self.contrastive_model,
                                                                         self.train_dataloader, self.valid_dataloader,
                                                                         self.test_dataloader,
                                                                         self.test_logger, self.num_classes_target,
                                                                         device=self.device,
                                                                         model_path=self.model_path + "_classifier")
        classifier_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = classifier_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Contrastive Training", latent_embedding

    def _load_contrastive_model(self):
        base_path = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}"
        contrastive_state = torch.load(
            base_path + "/model_weights/" + self.model_path + "_contrastive" + "/final_contrastive_model_state.pth")
        self.contrastive_model.load_state_dict(contrastive_state["model"])
        self.contrastive_model = self.contrastive_model.to(self.device)
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

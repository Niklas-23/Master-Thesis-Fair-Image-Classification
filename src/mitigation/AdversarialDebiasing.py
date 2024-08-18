import torch
from torch import nn

from src.datasets.dataset_wrapper import DatasetWrapper
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer


class AdversarialDebiasingBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32, fairness_constraint="eod", alpha=1):
        super().__init__(model, dataset, model_path, device, batch_size)
        self.fairness_constraint = fairness_constraint
        self.alpha = alpha

    def apply_pre_processing(self):
        self.logger.info("Start adversairal debiasing benchmark.")
        pass

    def train_model(self):
        model_trainer = AdversarialDebiasingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                         self.test_dataloader,
                                                         self.logger,
                                                         device=self.device,
                                                         model_path=self.model_path,
                                                         num_classes_target=self.num_classes_target,
                                                         num_classes_protected_feature=self.num_classes_protected_feature,
                                                         fairness_constraint=self.fairness_constraint,
                                                         alpha=self.alpha)
        model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        test_model_trainer = AdversarialDebiasingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                         self.test_dataloader,
                                                         self.test_logger,
                                                         device=self.device,
                                                         model_path=self.model_path,
                                                         num_classes_target=self.num_classes_target,
                                                         num_classes_protected_feature=self.num_classes_protected_feature,
                                                         fairness_constraint=self.fairness_constraint,
                                                         alpha=self.alpha)
        test_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = test_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Adversarial Debiasing: {test_results}")
        return "Adversarial Debiasing", test_results

    def generate_latent_embedding(self, model_state_version: str = None):
        test_model_trainer = AdversarialDebiasingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                              self.test_dataloader,
                                                              self.test_logger,
                                                              device=self.device,
                                                              model_path=self.model_path,
                                                              num_classes_target=self.num_classes_target,
                                                              num_classes_protected_feature=self.num_classes_protected_feature,
                                                              fairness_constraint=self.fairness_constraint,
                                                              alpha=self.alpha)
        test_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = test_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Adversarial Debiasing", latent_embedding


class AdversarialModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AdversarialModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class AdversarialDebiasingModelTrainer(ModelTrainer):

    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, device, model_path, num_classes_target,
                 num_classes_protected_feature, fairness_constraint, alpha):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target, device, model_path)

        self.fairness_constraint = fairness_constraint
        self.alpha = alpha
        self.num_classes_target = num_classes_target
        if fairness_constraint == "eod":
            input_size = 2 * num_classes_target
        else:
            input_size = num_classes_target
        self.adversary_model = AdversarialModel(input_size=input_size, output_size=num_classes_protected_feature).to(device)
        self.predictor_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.adversary_optimizer = torch.optim.Adam(self.adversary_model.parameters(), lr=0.001)
        self.adversary_loss = nn.CrossEntropyLoss()
        self.loss_function = nn.CrossEntropyLoss()

    def train_step(self, images, targets, protected_features, epoch):
        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()
        targets = targets.long()
        protected_features = protected_features.long()
        Y_hat = self.model(images)
        Y_hat_return = Y_hat.clone()
        LP = self.loss_function(Y_hat, targets)
        LP.backward(retain_graph=True)

        dW_LP = [
            torch.clone(p.grad.detach()) for p in self.model.parameters()
        ]

        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        # For equalized odds
        if self.fairness_constraint == "eod":
            target_one_hot_encoded = torch.nn.functional.one_hot(targets, num_classes=self.num_classes_target)
            Y_hat = torch.cat((Y_hat, target_one_hot_encoded), dim=1)

        A_hat = self.adversary_model(Y_hat)
        LA = self.adversary_loss(A_hat, protected_features)
        LA.backward()

        dW_LA = [
            torch.clone(p.grad.detach()) for p in self.model.parameters()
        ]

        for i, p in enumerate(self.model.parameters()):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
            # Project
            proj = torch.sum(unit_dW_LA * dW_LP[i])
            # Calculate dW
            p.grad = dW_LP[i] - (proj * unit_dW_LA) - (self.alpha * dW_LA[i])

        self.predictor_optimizer.step()
        self.adversary_optimizer.step()

        return Y_hat_return, LP

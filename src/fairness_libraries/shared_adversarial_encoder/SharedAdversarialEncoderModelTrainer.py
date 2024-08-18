import torch

from src.model_training import ModelTrainer

ADVERSARIAL_LAMBDA = 1


class SharedAdversarialEncoderModelTrainer(ModelTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target, device="cpu", model_path=None):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target, device=device, model_path=model_path)

        classification_optimizer_params = list(model.resnet.parameters()) + list(model.classifier_head.parameters())
        self.optimizer = torch.optim.Adam(classification_optimizer_params, lr=0.001)
        self.adversarial_optimizer = torch.optim.Adam(self.model.adversarial_head.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()
        y, a, a_detach = self.model(images, get_adv=True)
        classification_loss = self.criterion(y, targets.long())
        adversarial_loss = self.criterion(a, protected_features.long())
        train_loss = classification_loss - ADVERSARIAL_LAMBDA * adversarial_loss
        train_loss.backward()
        self.optimizer.step()

        self.adversarial_optimizer.zero_grad()
        adversarial_train_loss = self.criterion(a_detach, protected_features.long())
        adversarial_train_loss.backward()
        self.adversarial_optimizer.step()

        return y, train_loss

from src.datasets.dataset_wrapper import DatasetWrapper
from src.fairness_libraries.entangling_disentangling_bias.EnD import EnDLoss
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer


class EntanglingDisentanglingBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)

    def apply_pre_processing(self):
        self.logger.info("Start entangling disentangling benchmark.")
        pass

    def train_model(self):
        model_trainer = EntanglingDisentanglingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.logger, self.num_classes_target, device=self.device,
                                     model_path=self.model_path)
        model_trainer.set_loss_function(EnDLoss(self.model))
        model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        end_model_trainer = EntanglingDisentanglingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                            self.test_dataloader,
                                                            self.test_logger, self.num_classes_target, device=self.device,
                                                            model_path=self.model_path)

        end_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = end_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results EnD: {test_results}")
        return "Entangling Disentangling", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        end_model_trainer = EntanglingDisentanglingModelTrainer(self.model, self.train_dataloader,
                                                                self.valid_dataloader,
                                                                self.test_dataloader,
                                                                self.test_logger, self.num_classes_target,
                                                                device=self.device,
                                                                model_path=self.model_path)

        end_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = end_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Entangling Disentangling", latent_embedding


class EntanglingDisentanglingModelTrainer(ModelTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                 device="cpu", model_path=None):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                         device=device, model_path=model_path)

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()
        outputs = self.model(images)

        loss, abs = self.loss_function(outputs, targets.float(), protected_features)
        train_loss = loss+abs

        train_loss.backward()
        self.optimizer.step()

        return outputs, train_loss

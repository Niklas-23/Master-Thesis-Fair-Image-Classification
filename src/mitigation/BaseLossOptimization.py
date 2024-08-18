from src.datasets.dataset_wrapper import DatasetWrapper
from src.fairness_libraries.base_loss.BiasLogitSigmoidVarLoss import BiasLogitSigmoidVarLoss
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer


class BaseLossOptimizationBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)

    def apply_pre_processing(self):
        self.logger.info("Start base loss benchmark.")
        pass

    def train_model(self):
        base_model_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.logger, self.num_classes_target, device=self.device, model_path=self.model_path)
        base_model_trainer.set_loss_function(BiasLogitSigmoidVarLoss(num_classes=self.num_classes_target))
        base_model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        base_model_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                        self.test_logger, self.num_classes_target, device=self.device,
                                        model_path=self.model_path)
        base_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = base_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results BASE Loss: {test_results}")
        return "BASE Loss", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        base_model_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                          self.test_dataloader,
                                          self.test_logger, self.num_classes_target, device=self.device,
                                          model_path=self.model_path)
        base_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = base_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "BASE Loss", latent_embedding

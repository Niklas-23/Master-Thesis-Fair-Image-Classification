from torch.utils.data import DataLoader

from src.datasets.dataset_wrapper import DatasetWrapper
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer




class BaselineBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)
        train_dataset, valid_dataset, test_dataset = self.dataset.get_datasets()
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    def apply_pre_processing(self):
        pass

    def train_model(self):
        baseline_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.logger, self.num_classes_target, device=self.device, model_path=self.model_path)
        baseline_trainer.train()
        pass

    def test_model(self,  model_state_version: str = None):
        baseline_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                        self.test_logger, self.num_classes_target, device=self.device,
                                        model_path=self.model_path)
        baseline_trainer.load_model(extra_model_version=model_state_version)
        test_results = baseline_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Baseline: {test_results}")
        return "Baseline", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        baseline_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                        self.test_logger, self.num_classes_target, device=self.device,
                                        model_path=self.model_path)
        baseline_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = baseline_trainer.generate_latent_embedding(self.test_dataloader)
        return "Baseline", latent_embedding

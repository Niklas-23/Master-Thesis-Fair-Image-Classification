from src.datasets.dataset_wrapper import DatasetWrapper
from src.fairness_libraries.shared_adversarial_encoder.SharedAdversarialEncoderModelTrainer import \
    SharedAdversarialEncoderModelTrainer
from src.mitigation.benchmark_interface import BenchmarkInterface


class SharedAdversarialEncoderBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)

    def apply_pre_processing(self):
        self.logger.info("Start shared adversarial encoder benchmark.")
        pass

    def train_model(self):
        model_trainer = SharedAdversarialEncoderModelTrainer(self.model,  self.train_dataloader,
                                     self.valid_dataloader,
                                     self.test_dataloader,
                                     self.logger,
                                     self.num_classes_target,
                                     device=self.device,
                                     model_path=self.model_path)
        model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        shared_adv_model_trainer = SharedAdversarialEncoderModelTrainer(self.model, self.train_dataloader,
                                                             self.valid_dataloader,
                                                             self.test_dataloader,
                                                             self.test_logger,
                                                             self.num_classes_target,
                                                             device=self.device,
                                                             model_path=self.model_path)
        shared_adv_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = shared_adv_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Adversarial Shared Encoder: {test_results}")
        return "Adversarial Shared Encoder", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        shared_adv_model_trainer = SharedAdversarialEncoderModelTrainer(self.model, self.train_dataloader,
                                                                        self.valid_dataloader,
                                                                        self.test_dataloader,
                                                                        self.test_logger,
                                                                        self.num_classes_target,
                                                                        device=self.device,
                                                                        model_path=self.model_path)
        shared_adv_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = shared_adv_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Adversarial Shared Encoder", latent_embedding




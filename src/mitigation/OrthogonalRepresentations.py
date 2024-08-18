from src.datasets.dataset_wrapper import DatasetWrapper
from src.fairness_libraries.fcro.FCRO_model import ResNet18FCRO
from src.fairness_libraries.fcro.FCRO_trainer import FCROTrainer
from src.mitigation.benchmark_interface import BenchmarkInterface


class OrthogonalRepresentationsBenchmark(BenchmarkInterface):
    def __init__(self, model_t, model_a, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(None, dataset, model_path, device, batch_size)
        self.model_t = model_t
        self.model_a = model_a

    def apply_pre_processing(self):
        self.logger.info("Start orthogonal representation benchmark.")
        pass

    def train_model(self):
        self.logger.info("Train model protected feature")
        model_trainer_a = FCROTrainer(self.model_a, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                      logger=self.logger,
                                      num_classes_target=self.num_classes_target,
                                      num_classes_protected_feature=self.num_classes_protected_feature,
                                      model_t=self.model_t,
                                      model_a=self.model_a,
                                      mode=0,
                                      batch_size=self.batch_size,
                                      device=self.device,
                                      model_path=self.model_path+"_protected")

        model_trainer_a.train()
        self.logger.info("Train model target")
        model_trainer_t = FCROTrainer(self.model_t, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                      logger=self.logger,
                                      num_classes_target=self.num_classes_target,
                                      num_classes_protected_feature=self.num_classes_protected_feature,
                                      model_t=self.model_t,
                                      model_a=self.model_a,
                                      mode=1,
                                      batch_size=self.batch_size,
                                      device=self.device,
                                      model_path=self.model_path+"_target")
        model_trainer_t.train()
        pass

    def test_model(self, model_state_version: str = None):
        model_trainer_t = FCROTrainer(self.model_t, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                      logger=self.test_logger,
                                      num_classes_target=self.num_classes_target,
                                      num_classes_protected_feature=self.num_classes_protected_feature,
                                      model_t=self.model_t,
                                      model_a=self.model_a,
                                      mode=1,
                                      batch_size=self.batch_size,
                                      device=self.device,
                                      model_path=self.model_path + "_target")
        model_trainer_t.load_model(extra_model_version=model_state_version)
        test_results = model_trainer_t.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Orthogonal Representations: {test_results}")
        return "Orthogonal Representations", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        model_trainer_t = FCROTrainer(self.model_t, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                      logger=self.test_logger,
                                      num_classes_target=self.num_classes_target,
                                      num_classes_protected_feature=self.num_classes_protected_feature,
                                      model_t=self.model_t,
                                      model_a=self.model_a,
                                      mode=1,
                                      batch_size=self.batch_size,
                                      device=self.device,
                                      model_path=self.model_path + "_target")
        model_trainer_t.load_model(extra_model_version=model_state_version)
        latent_embedding = model_trainer_t.generate_latent_embedding(self.test_dataloader)
        return "Adversarial Shared Encoder", latent_embedding

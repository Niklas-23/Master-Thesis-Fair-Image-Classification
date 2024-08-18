from src.constants import UTK_FACE_DATASET, FAIR_FACE_DATASET
from src.datasets.dataset_wrapper import DatasetWrapper
from src.fairness_libraries.fair_feature_distillation.FairFeatureDistillationModelTrainer import \
    FairFeatureDistillationModelTrainer
from src.mitigation.benchmark_interface import BenchmarkInterface


class FairFeatureDistillationBenchmark(BenchmarkInterface):
    def __init__(self, model, teacher, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)
        self.teacher = teacher.to(device)
        if dataset.dataset_name == UTK_FACE_DATASET or dataset.dataset_name == FAIR_FACE_DATASET:
            self.lambda_feature_distill = 3
        else:
            self.lambda_feature_distill = 7

    def apply_pre_processing(self):
        self.logger.info("Start fair feature distillation benchmark.")
        pass

    def train_model(self):
        model_trainer = FairFeatureDistillationModelTrainer(self.model, self.teacher,  self.train_dataloader,
                                                            self.valid_dataloader,
                                                            self.test_dataloader,
                                                            self.logger,
                                                            self.num_classes_target,
                                                            self.num_classes_protected_feature,
                                                            device=self.device,
                                                            model_path=self.model_path,
                                                            lambda_feature_distill=self.lambda_feature_distill)
        model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        feat_distill_model_trainer = FairFeatureDistillationModelTrainer(self.model, self.teacher, self.train_dataloader,
                                                            self.valid_dataloader,
                                                            self.test_dataloader,
                                                            self.test_logger,
                                                            self.num_classes_target,
                                                            self.num_classes_protected_feature,
                                                            device=self.device,
                                                            model_path=self.model_path,
                                                            lambda_feature_distill=self.lambda_feature_distill)

        feat_distill_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = feat_distill_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Feature Distillation: {test_results}")
        return "Feature Distillation", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        feat_distill_model_trainer = FairFeatureDistillationModelTrainer(self.model, self.teacher,
                                                                         self.train_dataloader,
                                                                         self.valid_dataloader,
                                                                         self.test_dataloader,
                                                                         self.test_logger,
                                                                         self.num_classes_target,
                                                                         self.num_classes_protected_feature,
                                                                         device=self.device,
                                                                         model_path=self.model_path,
                                                                         lambda_feature_distill=self.lambda_feature_distill)

        feat_distill_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding = feat_distill_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Feature Distillation", latent_embedding

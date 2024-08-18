import numpy as np

from src.datasets.dataset_wrapper import DatasetWrapper
from src.fairness_libraries.blackbox_multiclass_fairness.MulticlassBalancer import BinaryBalancer, MulticlassBalancer
from src.fairness_libraries.fair_feature_distillation.FairFeatureDistillationModelTrainer import \
    FairFeatureDistillationModelTrainer
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer


class BlackboxMulticlassBalancerBechmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, baseline_model_path:str,  device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)
        self.baseline_model_path = baseline_model_path

    def apply_pre_processing(self):
        self.logger.info("Start blackbox multiclass balancer benchmark.")
        pass

    def train_model(self):
        model_trainer = ModelTrainer(self.model,  self.train_dataloader,
                                     self.valid_dataloader,
                                     self.test_dataloader,
                                     self.logger,
                                     self.num_classes_target,
                                     device=self.device,
                                     model_path=self.model_path,
                                     post_processing_function=self.apply_post_processing)
        model_trainer.load_model(extra_model_path=self.baseline_model_path)
        metrics = model_trainer.validate_model(self.valid_dataloader, mode="valid")
        self.logger.info(f"Validation metrics: {metrics}")
        pass

    def test_model(self, model_state_version: str = None):
        blackbox_model_trainer = ModelTrainer(self.model, self.train_dataloader,
                                     self.valid_dataloader,
                                     self.test_dataloader,
                                     self.test_logger,
                                     self.num_classes_target,
                                     device=self.device,
                                     model_path=self.model_path,
                                     post_processing_function=self.apply_post_processing)
        blackbox_model_trainer.load_model(extra_model_path=self.baseline_model_path)
        test_results = blackbox_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Blackbox Postprocessing: {test_results}")
        return "Blackbox Postprocessing", test_results

    def apply_post_processing(self, y_true, y_pred, protected_features):
        y_pred = [int(x) for x in y_pred]
        y_true = [int(x) for x in y_true]
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        protected_features = np.array(protected_features)
        if self.dataset.get_num_classes_target() == 2:
            balancer = BinaryBalancer(y_true, y_pred, protected_features, summary=False)
            balancer.adjust(goal="odds", summary=False)
        else:
            balancer = MulticlassBalancer(y_true, y_pred, protected_features, summary=False)
            balancer.adjust(goal="odds", cv=True, summary=False)

        y_hat_pred = balancer.predict(y_pred, protected_features)
        return y_hat_pred


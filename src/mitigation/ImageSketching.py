from tqdm import tqdm

from src.dataset_configs import get_dataset_config
from src.datasets.dataset_wrapper import DatasetWrapper
from src.evaluation_metrics import EvaluationMetrics
from src.fairness_libraries.image_sketching_loss.ImageSketchingLoss import ImageSketchingLoss
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer
from src.util.image_sketching import create_sketch_images


class ImageSketchingBenchmark(BenchmarkInterface):

    def __init__(self, model, dataset: DatasetWrapper, model_path="image_sketching/default", device="cpu",
                 batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)

    def apply_pre_processing(self):
        self.logger.info("Start image sketching benchmark benchmark.")
        dataset_directory, image_directory = get_dataset_config(self.dataset.dataset_name)
        create_sketch_images(dataset_directory, image_directory)
        pass

    def train_model(self):
        model_trainer = ImageSketchingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     device=self.device, model_path=self.model_path, logger=self.logger, num_classes_target=self.num_classes_target)
        model_trainer.set_loss_function(ImageSketchingLoss())
        model_trainer.train()
        pass

    def test_model(self, model_state_version: str = None):
        sketching_model_trainer = ImageSketchingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                   self.test_dataloader,
                                                   device=self.device, model_path=self.model_path, logger=self.test_logger,
                                                   num_classes_target=self.num_classes_target)
        sketching_model_trainer.set_loss_function(ImageSketchingLoss())
        sketching_model_trainer.load_model(extra_model_version=model_state_version)
        test_results = sketching_model_trainer.validate_model(self.test_dataloader, mode="test")
        self.test_logger.info(f"Test results Image Sketching: {test_results}")
        return "Image Sketching", test_results

    def generate_latent_embedding(self,  model_state_version: str = None):
        sketching_model_trainer = ImageSketchingModelTrainer(self.model, self.train_dataloader, self.valid_dataloader,
                                                             self.test_dataloader,
                                                             device=self.device, model_path=self.model_path,
                                                             logger=self.test_logger,
                                                             num_classes_target=self.num_classes_target)
        sketching_model_trainer.set_loss_function(ImageSketchingLoss())
        sketching_model_trainer.load_model(extra_model_version=model_state_version)
        latent_embedding =  sketching_model_trainer.generate_latent_embedding(self.test_dataloader)
        return "Image Sketching", latent_embedding


class ImageSketchingModelTrainer(ModelTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                 device="cpu", model_path=None):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger, num_classes_target,
                         device=device, model_path=model_path)


    def train(self):

        best_valid_f1 = 0
        best_valid_eod = float("inf")
        best_loss = float("inf")

        train_y_true = []
        train_y_pred = []
        train_y_logit = []
        train_y_protected_feature = []

        self.logger.info("Training started.")
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            self.model.train()
            self.loss_function.reset_spd_list()
            for images, targets, protected_features in tqdm(self.train_dataloader):
                images, targets, protected_features = self._batch_to_device(images, targets, protected_features)

                self.logger.debug(f"------------ EPOCH {epoch} ------------")
                self.logger.debug(f"IMAGES: \n {images}")
                self.logger.debug(f"TARGETS: \n {targets}")
                self.logger.debug(f"PROTECTED FEATURES: \n {protected_features}")
                self.logger.debug(f"SHAPES (images, targets): {images.size()}, {targets.size()}")

                outputs, train_loss = self.train_step(images, targets, protected_features, epoch)

                train_y_logit.extend(outputs.detach().cpu().numpy())
                _, y_pred = outputs.max(dim=1)
                train_y_pred.extend(y_pred.detach().cpu().numpy())
                train_y_true.extend(targets.detach().cpu().numpy())
                train_y_protected_feature.extend(protected_features.detach().cpu().numpy())

                self.logger.debug(f"OUTPUTS: \n {outputs}")
                self.logger.debug(f"SHAPES outputs: {outputs.size()}")
                self.logger.debug(f"LOSS: {train_loss}")

                running_loss += train_loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_dataloader.dataset)

            # Log train process

            metrics = EvaluationMetrics(train_y_true, train_y_pred, train_y_logit, train_y_protected_feature,
                                        logger=self.logger)

            if self.num_classes_target > 2:
                eod_max, eod_mean = metrics.get_equalized_odds_difference()
                dp_max, dp_mean = metrics.get_demographic_parity_difference()
                train_progress_metrics = {
                    f"train_acc": metrics.get_accuracy(),
                    f"train_f1": metrics.get_f1(),
                    f"train_eod_max": eod_max,
                    f"train_eod_mean": eod_mean,
                    f"train_eod_strict": metrics.get_strict_multiclass_equalized_odds(),
                    f"train_dp_max": dp_max,
                    f"train_dp_mean": dp_mean,
                    "train_loss": epoch_loss
                }
                self.writer.add_scalar('EOD_max/train', train_progress_metrics["train_eod_max"], epoch)
                self.writer.add_scalar('EOD_mean/train', train_progress_metrics["train_eod_mean"], epoch)
                self.writer.add_scalar('EOD_strict/train', train_progress_metrics["train_eod_strict"], epoch)
                self.writer.add_scalar('DP_max/train', train_progress_metrics["train_dp_max"], epoch)
                self.writer.add_scalar('DP_mean/train', train_progress_metrics["train_dp_mean"], epoch)
            else:
                train_progress_metrics = {
                    f"train_acc": metrics.get_accuracy(),
                    f"train_f1": metrics.get_f1(),
                    f"train_eod": metrics.get_equalized_odds_difference(),
                    f"train_dp": metrics.get_demographic_parity_difference(),
                    "train_loss": epoch_loss
                }
                self.writer.add_scalar('EOD/train', train_progress_metrics["train_eod"], epoch)
                self.writer.add_scalar('DP/train', train_progress_metrics["train_dp"], epoch)

            self.train_process.append(train_progress_metrics)

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_progress_metrics["train_acc"], epoch)
            self.writer.add_scalar('F1/train', train_progress_metrics["train_f1"], epoch)

            # Log validation process
            valid_metrics = self.validate_model(self.valid_dataloader, "valid")
            self.valid_process.append(valid_metrics)
            valid_acc = valid_metrics["valid_acc"]
            valid_f1 = valid_metrics["valid_f1"]

            # Log to tensorboard
            self.writer.add_scalar('Accuracy/validation', valid_acc, epoch)
            self.writer.add_scalar('F1/validation', valid_f1, epoch)

            if self.num_classes_target > 2:
                valid_eod = valid_metrics["valid_eod_max"]
                valid_dp = valid_metrics["valid_dp_max"]
                self.writer.add_scalar('EOD_max/validation', valid_metrics["valid_eod_max"], epoch)
                self.writer.add_scalar('EOD_mean/validation', valid_metrics["valid_eod_mean"], epoch)
                self.writer.add_scalar('EOD_strict/validation', valid_metrics["valid_eod_strict"], epoch)
                self.writer.add_scalar('DP_max/validation', valid_metrics["valid_dp_max"], epoch)
                self.writer.add_scalar('DP_mean/validation', valid_metrics["valid_dp_mean"], epoch)
            else:
                valid_eod = valid_metrics["valid_eod"]
                valid_dp = valid_metrics["valid_dp"]
                self.writer.add_scalar('EOD/validation', valid_eod, epoch)
                self.writer.add_scalar('DP/validation', valid_dp, epoch)

            self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}, Valid DP: {valid_dp:.4f}, Valid EOD: {valid_eod:.4f}")

            # Save best f1 model
            if best_valid_f1 < valid_f1 and valid_f1 != 0:
                best_valid_f1 = valid_f1
                self.save_model("best_f1", epoch)

            # Save best eod model
            if best_valid_eod > valid_eod and valid_eod != 0:
                best_valid_eod = valid_eod
                self.save_model("best_eod", epoch)

            # Save best loss model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                self.save_model("best_loss", epoch)

            self.current_epoch += 1

        self.save_model_training()
        self.logger.info("Training finished.")
        pass
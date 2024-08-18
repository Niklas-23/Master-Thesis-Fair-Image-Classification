from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from src.datasets.dataset_wrapper import DatasetWrapper
from src.util.logging import create_logger


class BenchmarkInterface(ABC):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32, ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model_path = model_path
        self.batch_size = batch_size

        train_dataset, valid_dataset, test_dataset = self.dataset.get_datasets()

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        self.num_classes_target = dataset.get_num_classes_target()
        self.num_classes_protected_feature = dataset.get_num_classes_protected_feature()
        #self.logger = create_logger(name=model_path)
        self.test_logger = create_logger(name=model_path, filename="test_benchmark")
        pass

    @abstractmethod
    def apply_pre_processing(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self,  model_state_version: str = None):
        pass

    def run_benchmark(self):
        self.apply_pre_processing()
        self.train_model()

from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import Dataset


class DatasetWrapper(ABC):

    @abstractmethod
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        return None, None, None

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    @property
    @abstractmethod
    def get_num_classes_protected_feature(self):
        pass

    @property
    @abstractmethod
    def get_num_classes_target(self):
        pass

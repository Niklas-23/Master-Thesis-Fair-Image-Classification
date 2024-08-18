import os
from typing import List, Tuple, Union

import pandas as pd
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CelebA
from torchvision.transforms import transforms

from src.constants import CELEBA_DATASET
from src.datasets.dataset_wrapper import DatasetWrapper


def get_celeba_protected_feature_index(bias_label_name: str):
    if bias_label_name == "gender":
        return 20
    else:
        raise Exception("Bias label not valid")


class CustomCelebaDataset(Dataset):
    def __init__(self, celeba_dataset: Union[CelebA, Subset], target_name: int, protected_feature_name: int):
        self.samples = celeba_dataset

        self.selected_target_index = target_name
        self.protected_feature_index = protected_feature_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #target = torch.eye(2)[self.samples[idx][1][self.selected_target_index]].squeeze()
        #bias = torch.eye(2)[self.samples[idx][1][self.protected_feature_index]].squeeze()

        return self.samples[idx][0],  self.samples[idx][1][self.selected_target_index],  self.samples[idx][1][self.protected_feature_index]

    def get_dataframe(self):
        attr_names: List[str] = [name for name in self.samples.attr_names if name.strip()]
        data = []
        for sample in tqdm(self.samples):
            data.append(sample[1].tolist())
        df = pd.DataFrame(data, columns=attr_names)

        return df


class CelebADataset(DatasetWrapper):

    def __init__(self, input_shape=(128, 128), usage: float = 1, transform=None, target_name: int = 31, protected_feature_name: int = 20):
        self._dataset_name = CELEBA_DATASET

        current_file_path = os.path.abspath(__file__)
        dataset_path = os.path.join(os.path.dirname(current_file_path), '../../datasets_data')

        if transform is None:
            celeba_transform = transforms.Compose([
                transforms.Resize(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000))
            ])
        else:
            celeba_transform = transform

        self.train_dataset: CelebA = torchvision.datasets.CelebA(root=dataset_path, split="train",
                                                                 transform=celeba_transform,
                                                                 download=True)
        self.validation_dataset: CelebA = torchvision.datasets.CelebA(root=dataset_path, split="valid",
                                                                      transform=celeba_transform, download=True)
        self.test_dataset: CelebA = torchvision.datasets.CelebA(root=dataset_path, split="test",
                                                                transform=celeba_transform,
                                                                download=True)

        self.attr_names: List[str] = [name for name in self.train_dataset.attr_names if name.strip()]
        self.selected_target_index: int = target_name #31 #Smiling
        self.protected_feature_index: int = protected_feature_name #20 #gender
        print(f"Target: {self.attr_names[self.selected_target_index]}")
        print(f"Protected feature: {self.attr_names[self.protected_feature_index]}")

        if usage < 1.0:
            self.train_dataset = Subset(self.train_dataset, range(round(len(self.train_dataset) * usage)))
            self.validation_dataset = Subset(self.validation_dataset,
                                             range(round(len(self.validation_dataset) * usage)))
            self.test_dataset = Subset(self.test_dataset, range(round(len(self.test_dataset) * usage)))

    def get_attribute_names(self):
        return self.attr_names

    @property
    def dataset_name(self):
        return self._dataset_name

    def get_datasets(self) -> Tuple[CustomCelebaDataset, CustomCelebaDataset, CustomCelebaDataset]:
        """
        return train_dataset, validation_dataset, test_dataset
        """
        return CustomCelebaDataset(self.train_dataset, self.selected_target_index, self.protected_feature_index), \
               CustomCelebaDataset(self.validation_dataset, self.selected_target_index, self.protected_feature_index), \
               CustomCelebaDataset(self.test_dataset, self.selected_target_index, self.protected_feature_index)

    def set_target_label(self, selected_attribute: str):
        if selected_attribute not in self.attr_names:
            raise Exception("Specified attribute doesn't exist")
        self.selected_target_index = self.attr_names.index(selected_attribute)
        print(
            f"Updated target label. Selected label {selected_attribute}. Corresponding index: {self.selected_target_index}")
        print("Remember to rerequest the dataset")
        pass

    def set_bias_label(self, bias_label_name: str):
        self.protected_feature_index = get_celeba_protected_feature_index(bias_label_name)
        print(
            f"Updated bias label. Selected label {bias_label_name}. Corresponding index: {self.protected_feature_index}")
        print("Remember to rerequest the dataset")
        pass

    def get_num_classes_protected_feature(self):
        return 2

    def get_num_classes_target(self):
        return 2

    def get_target_protected_feature_names(self):
        return self.selected_target_index, self.protected_feature_index
import os
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

from src.constants import LFW_DATASET
from src.datasets.dataset_wrapper import DatasetWrapper
from PIL import Image


class CustomLFWDataset(Dataset):
    def __init__(self, dataset_path:str, samples_dataframe, transform):
        self.samples_dataframe = samples_dataframe.reset_index()
        self.image_paths = [f"{dataset_path}/lfw-deepfunneled/{file_name}" for file_name in samples_dataframe["filename_complete"].tolist()]
        self.transform = transform

    def __len__(self):
        return self.samples_dataframe.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image_tensor = self.transform(image)

        return image_tensor, torch.tensor(self.samples_dataframe.at[idx, "target_encoded"]), torch.tensor(
            self.samples_dataframe.at[idx, "protected_feature_encoded"])


class LFWDataset(DatasetWrapper):

    def __init__(self, input_shape=(128, 128), target_name: str = "Smiling", protected_feature_name: str = "male_gender", transform=None, csv_name=None):
        self._dataset_name = LFW_DATASET
        current_file_path = os.path.abspath(__file__)
        self.dataset_path = os.path.join(os.path.dirname(current_file_path), "../../datasets_data/LFW")
        if csv_name is None:
            attributes_dataframe = pd.read_csv(f"{self.dataset_path}/lfw_dataset.csv")
        else:
            attributes_dataframe = pd.read_csv(f"{self.dataset_path}/{csv_name}.csv")

        if protected_feature_name == "male_gender":
            attributes_dataframe["protected_feature_encoded"] = attributes_dataframe["male_gender"]
        else:
            attributes_dataframe["protected_feature_encoded"] = attributes_dataframe[protected_feature_name]

        attributes_dataframe["target_encoded"] = attributes_dataframe[target_name]

        self.train_dataframe, combined_dataframe = train_test_split(attributes_dataframe, test_size=0.3, random_state=42)
        self.valid_dataframe, self.test_dataframe = train_test_split(combined_dataframe, test_size=0.6, random_state=42)

        self.attr_names: List[str] = attributes_dataframe.columns.tolist()
        self.selected_target_name = target_name
        self.protected_feature_name = protected_feature_name

        if transform is None:
            self.lfw_transform = transforms.Compose([
                transforms.Resize(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000))
            ])
        else:
            self.lfw_transform = transform

    def get_attribute_names(self):
        return self.attr_names

    @property
    def dataset_name(self):
        return self._dataset_name

    def get_datasets(self) -> Tuple[CustomLFWDataset, CustomLFWDataset, CustomLFWDataset]:
        """
        return train_dataset, validation_dataset, test_dataset
        """

        return CustomLFWDataset(self.dataset_path, self.train_dataframe, self.lfw_transform), \
               CustomLFWDataset(self.dataset_path, self.valid_dataframe, self.lfw_transform), \
               CustomLFWDataset(self.dataset_path, self.test_dataframe, self.lfw_transform)

    def get_train_dataframe(self):
        return self.train_dataframe

    def get_valid_dataframe(self):
        return self.train_dataframe

    def get_test_dataframe(self):
        return self.test_dataframe

    def _get_binary_encoding_from_feature(self, dataframe, column_name):
        encoded_feature = []
        for index, row in dataframe.iterrows():
            if row[column_name] < 0:
                encoded_feature.append(0)
            else:
                encoded_feature.append(1)
        return encoded_feature

    def get_num_classes_protected_feature(self):
        return self.train_dataframe["protected_feature_encoded"].nunique()

    def get_num_classes_target(self):
        return self.train_dataframe["target_encoded"].nunique()

    def get_target_protected_feature_names(self):
        return self.selected_target_name, self.protected_feature_name

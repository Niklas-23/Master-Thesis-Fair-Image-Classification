import os
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

from src.constants import UTK_FACE_DATASET
from src.datasets.dataset_wrapper import DatasetWrapper
from PIL import Image


class CustomUTKFaceDataset(Dataset):
    def __init__(self, dataset_path:str, samples_dataframe, transform):
        self.samples_dataframe = samples_dataframe.reset_index()
        self.image_paths = [f"{dataset_path}/UTKFace/{file_name}" for file_name in samples_dataframe["File"].tolist()]
        self.transform = transform

    def __len__(self):
        return self.samples_dataframe.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image_tensor = self.transform(image)

        return image_tensor, torch.tensor(self.samples_dataframe.at[idx, "target_encoded"]), torch.tensor(
            self.samples_dataframe.at[idx, "protected_feature_encoded"])


class UTKFaceDataset(DatasetWrapper):

    def __init__(self, input_shape=(128, 128), target_name: str = "Age", protected_feature_name: str = "Gender", transform = None, age_bins = None):
        self._dataset_name = UTK_FACE_DATASET
        current_file_path = os.path.abspath(__file__)
        self.dataset_path = os.path.join(os.path.dirname(current_file_path), "../../datasets_data/UTKFace")
        utk_dataframe = pd.read_csv(f"{self.dataset_path}/utk_face_extracted_info_cleaned.csv")

        if target_name == "Age":
            if age_bins == 5:
                utk_dataframe['target_encoded'], bins = pd.qcut(utk_dataframe['Age'], q=5, labels=False, retbins=True)
            elif age_bins == 10:
                utk_dataframe['target_encoded'], bins = pd.qcut(utk_dataframe['Age'], q=10, labels=False, retbins=True)
            elif age_bins is not None:
                utk_dataframe["target_encoded"] = pd.cut(utk_dataframe["Age"], age_bins, labels=range(len(age_bins)-1))
            else:
                utk_dataframe["target_encoded"] = pd.cut(utk_dataframe["Age"], [0,20,40,120], labels=range(3))
            #print(utk_dataframe["target_encoded"].value_counts())
        else:
            target_label_encoder = LabelEncoder()
            utk_dataframe["target_encoded"] = target_label_encoder.fit_transform(
                utk_dataframe[target_name])

        protected_feature_label_encoder = LabelEncoder()
        utk_dataframe["protected_feature_encoded"] = protected_feature_label_encoder.fit_transform(
                utk_dataframe[protected_feature_name])

        self.train_dataframe, combined_dataframe = train_test_split(utk_dataframe, test_size=0.3, random_state=42)
        self.valid_dataframe, self.test_dataframe = train_test_split(combined_dataframe, test_size=0.6, random_state=42)

        self.attr_names: List[str] = utk_dataframe.columns.tolist()
        self.selected_target_name = target_name
        self.protected_feature_name = protected_feature_name

        if transform is None:
            self.utk_face_transform = transforms.Compose([
                transforms.Resize(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000))
            ])
        else:
            self.utk_face_transform = transform

    def get_attribute_names(self):
        return self.attr_names

    @property
    def dataset_name(self):
        return self._dataset_name

    def get_datasets(self) -> Tuple[CustomUTKFaceDataset, CustomUTKFaceDataset, CustomUTKFaceDataset]:
        """
        return train_dataset, validation_dataset, test_dataset
        """

        return CustomUTKFaceDataset(self.dataset_path, self.train_dataframe, self.utk_face_transform), \
               CustomUTKFaceDataset(self.dataset_path, self.valid_dataframe, self.utk_face_transform), \
               CustomUTKFaceDataset(self.dataset_path, self.test_dataframe, self.utk_face_transform)

    def get_train_dataframe(self):
        return self.train_dataframe

    def get_valid_dataframe(self):
        return self.valid_dataframe

    def get_test_dataframe(self):
        return self.test_dataframe

    def get_num_classes_protected_feature(self):
        return self.train_dataframe["protected_feature_encoded"].nunique()

    def get_num_classes_target(self):
        return self.train_dataframe["target_encoded"].nunique()

    def get_target_protected_feature_names(self):
        return self.selected_target_name, self.protected_feature_name

import os
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

from src.constants import FAIR_FACE_DATASET
from src.datasets.dataset_wrapper import DatasetWrapper
from PIL import Image


class CustomFairFaceDataset(Dataset):
    def __init__(self, dataset_path: str, fair_face_label_csv, selected_target_name: str, protected_feature_name: str, transform):
        self.dataset_path = dataset_path
        self.fair_face_label_csv = fair_face_label_csv
        self.selected_target_name = selected_target_name
        self.protected_feature_name = protected_feature_name

        # one_hot_encoded_target = pd.get_dummies(fair_face_label_csv[selected_target_name])
        # one_hot_encoded_target = one_hot_encoded_target.map(lambda x: 1 if x else 0)
        # self.one_hot_target_tensors = [torch.tensor(row.values).float() for _, row in one_hot_encoded_target.iterrows()]

        # one_hot_encoded_bias = pd.get_dummies(fair_face_label_csv[bias_label_name])
        # one_hot_encoded_bias = one_hot_encoded_bias.map(lambda x: 1 if x else 0)
        # self.one_hot_bias_tensors = [torch.tensor(row.values).float() for _, row in one_hot_encoded_bias.iterrows()]

        target_label_encoder = LabelEncoder()
        self.fair_face_label_csv["target_encoded"] = target_label_encoder.fit_transform(fair_face_label_csv[selected_target_name])


        print(target_label_encoder.classes_)
        print(list(target_label_encoder.transform(["Female", "Male"])))

        protected_feature_label_encoder = LabelEncoder()
        self.fair_face_label_csv["protected_feature_encoded"] = protected_feature_label_encoder.fit_transform(fair_face_label_csv[protected_feature_name])
        print(protected_feature_label_encoder.classes_)
        print(list(protected_feature_label_encoder.transform(["Black", "White", "Indian"])))

        self.image_paths = [dataset_path+file_name for file_name in fair_face_label_csv["file"].tolist()]

        self.transform = transform

    def __len__(self):
        return self.fair_face_label_csv.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image_tensor = self.transform(image)

        # return image_tensor.float(), self.one_hot_target_tensors[idx], self.one_hot_bias_tensors[idx]
        return image_tensor, torch.tensor(self.fair_face_label_csv.at[idx, "target_encoded"]), torch.tensor(self.fair_face_label_csv.at[idx, "protected_feature_encoded"])


class FairFaceDataset(DatasetWrapper):

    def __init__(self, input_shape=(128, 128), target_name: str = "age", protected_feature_name: str = "race", transform=None):
        self._dataset_name = FAIR_FACE_DATASET
        self.target_name = target_name
        self.protected_feature_name = protected_feature_name

        current_file_path = os.path.abspath(__file__)
        self.dataset_path = os.path.join(os.path.dirname(current_file_path), '../../datasets_data/fairface-img-margin025-trainval/')

        if transform is None:
            self.fair_face_transform = transforms.Compose([
                transforms.Resize(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000))
            ])
        else:
            self.fair_face_transform = transform

        self.train_dataframe = pd.read_csv(f"{self.dataset_path}fairface_label_train.csv")
        self.test_dataframe = pd.read_csv(f"{self.dataset_path}fairface_label_val.csv")

        self.train_dataframe.loc[self.train_dataframe["age"].isin(
            ['40-49', '50-59', '60-69', 'more than 70']), "age"] = "40+"

        self.train_dataframe.loc[self.train_dataframe["age"].isin(
            ['0-2', '3-9', '10-19']), "age"] = "0-19"

        self.valid_dataframe = self.train_dataframe.sample(frac=0.1, random_state=42)
        self.train_dataframe = self.train_dataframe.drop(self.valid_dataframe.index)
        self.train_dataframe = self.train_dataframe.reset_index()
        self.valid_dataframe = self.valid_dataframe.reset_index()

        self.attr_names: List[str] = self.train_dataframe.columns.tolist()

    def get_attribute_names(self):
        return self.attr_names

    @property
    def dataset_name(self):
        return self._dataset_name

    def get_datasets(self) -> Tuple[CustomFairFaceDataset, CustomFairFaceDataset, CustomFairFaceDataset]:
        """
        return train_dataset, validation_dataset, test_dataset
        """

        return CustomFairFaceDataset(self.dataset_path, self.train_dataframe, self.target_name, self.protected_feature_name, self.fair_face_transform), \
               CustomFairFaceDataset(self.dataset_path, self.valid_dataframe, self.target_name, self.protected_feature_name, self.fair_face_transform), \
               CustomFairFaceDataset(self.dataset_path, self.test_dataframe, self.target_name, self.protected_feature_name, self.fair_face_transform)

    def get_train_dataframe(self):
        return self.train_dataframe

    def get_valid_dataframe(self):
        return self.valid_dataframe

    def get_test_dataframe(self):
        return self.test_dataframe

    def get_num_classes_protected_feature(self):
        return self.train_dataframe[self.protected_feature_name].nunique()

    def get_num_classes_target(self):
        return self.train_dataframe[self.target_name].nunique()

    def get_target_protected_feature_names(self):
        return self.target_name, self.protected_feature_name

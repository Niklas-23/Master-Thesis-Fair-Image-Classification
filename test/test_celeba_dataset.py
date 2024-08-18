import unittest
import pandas as pd
from tqdm import tqdm

from src.datasets.CelebA_dataset import CelebADataset


class TestCreateDataFrame(unittest.TestCase):

    def test_bias_labels(self):
        celeba_dataset = CelebADataset(input_shape=(128, 128), usage=0.1)
        train_dataset, valid_dataset, test_dataset = celeba_dataset.get_datasets()

        labels = []
        bias_labels = []
        for image, target, bias in tqdm(train_dataset):
            labels.append(target.squeeze().tolist())
            bias_labels.append(bias.item())

        df = pd.DataFrame(labels, columns=celeba_dataset.get_attribute_names())
        df_list = df['Male'].tolist()

        if bias_labels == df_list:
            print("list equal")
        else:
            print("list not equal")

        for i, e in enumerate(bias_labels):
            self.assertEqual(e, df_list[i])


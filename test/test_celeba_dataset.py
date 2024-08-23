import unittest
from tqdm import tqdm

from src.datasets.CelebA_dataset import CelebADataset


class TestCreateDataFrame(unittest.TestCase):

    def test_bias_labels(self):
        celeba_dataset = CelebADataset(input_shape=(128, 128))
        train_dataset, valid_dataset, test_dataset = celeba_dataset.get_datasets()

        bias_labels = []
        for image, target, bias in tqdm(train_dataset):
            bias_labels.append(bias.item())

        df_list = train_dataset.get_dataframe()['Male'].tolist()

        if bias_labels == df_list:
            print("list equal")
        else:
            print("list not equal")

        for i, e in enumerate(bias_labels):
            self.assertEqual(e, df_list[i])

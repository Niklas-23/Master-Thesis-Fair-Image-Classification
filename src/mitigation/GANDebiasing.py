from src.datasets.dataset_wrapper import DatasetWrapper
from src.mitigation.benchmark_interface import BenchmarkInterface
from src.model_training import ModelTrainer

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

class GANDebiasingBenchmark(BenchmarkInterface):
    def __init__(self, model, dataset: DatasetWrapper, model_path: str, device="cpu", batch_size=32):
        super().__init__(model, dataset, model_path, device, batch_size)

    def apply_pre_processing(self):
        # Randomly sample 512-dimensional latent vectors.
        z_t0g0 = np.random.normal(0.1, 1, size=(6000, 512))  # contains neither t or g
        z_t0g1 = np.random.normal(0.2, 1, size=(1000, 512))  # contains g, but not t
        z_t1g0 = np.random.normal(-0.1, 1, size=(1000, 512))  # contains t, but not g
        z_t1g1 = np.random.normal(-0.2, 1, size=(2000, 512))  # contains both t and g

        # Stack everything so that all properties can be indexed from the array.
        z = np.vstack((z_t0g0, z_t0g1, z_t1g0, z_t1g1))

        # Label the examples with whether they contain t.
        t = np.zeros(10000)
        t[:7000] = 0
        t[7000:] = 1

        # Label the examples with whether they contain g.
        g = np.zeros(10000)
        g[:6000] = 0
        g[6000:7000] = 1
        g[7000:8000] = 0
        g[8000:] = 1

        # Split the data into training and validation sets.
        train_indices = np.random.choice(10000, size=4000, replace=False)
        val_indices = np.random.choice(list(set(np.arange(10000)) - set(train_indices)), size=1000, replace=False)
        test_indices = np.array(list(set(np.arange(10000)) - set(train_indices) - set(val_indices)))

        z_train = z[train_indices]
        t_train = t[train_indices]
        g_train = g[train_indices]

        z_val = z[val_indices]
        t_val = t[val_indices]
        g_val = g[val_indices]

        z_test = z[test_indices]
        t_test = t[test_indices]
        g_test = g[test_indices]

        # Fit a target attribute classifier in the latent space using a linear SVM.
        h_t = svm.LinearSVC(max_iter=50000)
        h_t.fit(z_train, t_train)

        # Normalize so that w_t has norm 1.
        w_t_norm = np.linalg.norm(h_t.coef_)
        h_t.coef_ = h_t.coef_ / (w_t_norm)  # w_t
        h_t.intercept_ = h_t.intercept_ / w_t_norm  # b_t

        # Fit a protected attribute classifier in the latent space using a linear SVM.
        h_g = svm.LinearSVC(max_iter=50000)
        h_g.fit(z_train, g_train)

        # Normalize so that w_g has norm 1.
        w_g_norm = np.linalg.norm(h_g.coef_)
        h_g.coef_ = h_g.coef_ / (w_g_norm)
        h_g.intercept_ = h_g.intercept_ / w_g_norm

        # Run inference with h(z) = w^T z + b.
        t_val_prediction = np.sum(h_t.coef_ * z_val, axis=1) + h_t.intercept_
        g_val_prediction = np.sum(h_g.coef_ * z_val, axis=1) + h_g.intercept_

        # Calculate prediction accuracy
        t_val_correct = np.logical_and(t_val == 1, t_val_prediction >= 0).sum() + np.logical_and(t_val == 0,
                                                                                                 t_val_prediction < 0).sum()
        g_val_correct = np.logical_and(g_val == 1, g_val_prediction >= 0).sum() + np.logical_and(g_val == 0,
                                                                                                 g_val_prediction < 0).sum()
        self.logger.info('Accuracy of t classification: {}%'.format(t_val_correct / 1000 * 100))
        self.logger.info('Accuracy of g classification: {}%'.format(g_val_correct / 1000 * 100))

        # Calculate useful information.
        g_perp_t = h_g.coef_ - (np.sum(h_g.coef_ * h_t.coef_)) * h_t.coef_
        g_perp_t = g_perp_t / np.linalg.norm(g_perp_t)

        cos_theta = np.sum(h_g.coef_ * h_t.coef_)
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)  # cos(theta)^2 + sin(theta)^2 = 1

        # For every z, find z' with flipped protected attribute.
        z_prime = np.zeros((5000, 512))
        for j in range(5000):
            dist = np.sum(h_g.coef_ * z_test[j]) + h_g.intercept_  # w_g^T z_j + b_g
            z_prime[j] = z_test[j] - ((2 * dist) / sin_theta) * g_perp_t  # closed form solution for z'

        pass

    def train_model(self):
        model_trainer = ModelTrainer(self.model, self.train_dataloader, self.valid_dataloader, self.test_dataloader,
                                     self.num_classes_target, device=self.device, model_path=self.model_path)
        model_trainer.train()
        pass

    def test_model(self):
        pass

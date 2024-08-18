import argparse
import os
from datetime import datetime

import pandas as pd

from src.benchmark_models.ResNet18 import ResNet18
from src.datasets.CelebA_dataset import CelebADataset
from src.datasets.FairFaceDataset import FairFaceDataset
from src.datasets.LFWDataset import LFWDataset
from src.datasets.UTKFaceDataset import UTKFaceDataset
from src.mitigation.Baseline import BaselineBenchmark
from src.run_experiments import get_fair_feature_distillation_benchmark, get_orthogonal_representations_benchmark, \
    get_shared_adv_encoder_benchmark, get_fair_contrastive_learning_benchmark, get_entangling_disentangling_benchmark, \
    get_domain_independent_training_benchmark, get_blackbox_multiclass_balancer_benchmark, \
    get_base_loss_optimization_benchmark, get_adversarial_debiasing_benchmark, get_image_sketching_benchmark


def main(args):
    print("Start Testing")
    dataset_name = args.dataset_name
    model_path_version = args.model_path_version
    batch_size = args.batch_size
    device = args.device
    model_path = None
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    baseline_model_path = f'{current_file_path}/../model_weights/plain_baseline/'
    baseline_model_version = f"/{args.baseline_model_path}/best_f1_state.pth"
    experiment_dataset = None
    if dataset_name == "celeba":
        experiment_dataset = CelebADataset()
        model_path = f"celeba/{model_path_version}"
        baseline_model_path = baseline_model_path + "celeba" + baseline_model_version
    elif dataset_name == "utk":
        experiment_dataset = UTKFaceDataset()
        model_path = f"utk_face/{model_path_version}"
        baseline_model_path = baseline_model_path + "utk_face" + baseline_model_version
    elif dataset_name == "fair_face":
        experiment_dataset = FairFaceDataset(target_name="gender")
        model_path = f"fair_face/{model_path_version}"
        baseline_model_path = baseline_model_path + "fair_face" + baseline_model_version
    elif dataset_name == "lfw":
       # experiment_dataset = LFWDataset(target_name="Strong Nose-Mouth Lines", csv_name="lfw_dataset_male__strong_nose_mouth_skewed_0.99")
        experiment_dataset = LFWDataset(target_name="Wearing Lipstick")
        model_path = f"lfw/{model_path_version}"
        baseline_model_path = baseline_model_path + "lfw" + baseline_model_version

    mitigation_approaches = []
    mitigation_approaches.append(get_image_sketching_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(
         get_adversarial_debiasing_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(
         get_base_loss_optimization_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(
         get_blackbox_multiclass_balancer_benchmark(experiment_dataset, device, batch_size, model_path,
                                                   baseline_model_path))
    mitigation_approaches.append(
        get_domain_independent_training_benchmark(experiment_dataset, device, batch_size, model_path
                                                 ))
    mitigation_approaches.append(
         get_entangling_disentangling_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(
         get_fair_feature_distillation_benchmark(experiment_dataset, device, batch_size, model_path,
                                                 baseline_model_path))
    mitigation_approaches.append(get_orthogonal_representations_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(get_shared_adv_encoder_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(
       get_fair_contrastive_learning_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(
        get_baseline_benchmark(experiment_dataset, device, batch_size, model_path))

    for state_version in ["best_eod_state", "best_f1_state", "best_loss_state"]:
        results = []
        for approach in mitigation_approaches:
            name, metrics = approach.test_model(model_state_version=state_version)
            metrics['name'] = name
            results.append(metrics)
        df = pd.DataFrame(results)
        df.to_csv(f"{current_file_path}/../benchmark_test_results/{model_path}_{state_version}.csv", index=False)


def get_baseline_benchmark(dataset, device, batch_size, model_path):
    resnet18_model = ResNet18(num_classes=dataset.get_num_classes_target())
    return BaselineBenchmark(resnet18_model, dataset, device=device,
                                              model_path=f"plain_baseline/{model_path}",
                                              batch_size=batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', choices=['lfw', 'celeba', 'utk', 'fair_face'])
    parser.add_argument('--model_path_version', type=str, help='Version of the model path', default="v1")
    parser.add_argument('--baseline_model_path', type=str, help='Version of the model path')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=512)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', ',mps'],
                        help='Device for training (cpu/gpu/mps)')

    args = parser.parse_args()

    main(args)

import argparse

from src.benchmark_models.ResNet18 import ResNet18
from src.datasets.CelebA_dataset import CelebADataset
from src.datasets.FairFaceDataset import FairFaceDataset
from src.datasets.LFWDataset import LFWDataset
from src.datasets.UTKFaceDataset import UTKFaceDataset
from src.mitigation.Baseline import BaselineBenchmark


def main(args):
    dataset_name = args.dataset_name
    model_path_version = args.model_path_version
    batch_size = args.batch_size
    device = args.device
    model_path = None
    experiment_dataset = None
    if dataset_name == "celeba":
        experiment_dataset = CelebADataset()
        model_path = f"celeba/{model_path_version}"
    elif dataset_name == "utk":
        experiment_dataset = UTKFaceDataset()
        model_path = f"utk_face/{model_path_version}"
    elif dataset_name == "fair_face":
        experiment_dataset = FairFaceDataset(target_name="gender")
        model_path = f"fair_face/{model_path_version}"
    elif dataset_name == "lfw":
        experiment_dataset = LFWDataset(target_name="Wearing Lipstick")
        model_path = f"lfw/{model_path_version}"

    resnet18_model = ResNet18(num_classes=experiment_dataset.get_num_classes_target())
    baseline_benchmark = BaselineBenchmark(resnet18_model, experiment_dataset, device=device,
                                                   model_path=f"plain_baseline/{model_path}", batch_size=batch_size)
    baseline_benchmark.run_benchmark()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model baseline')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', choices=['lfw', 'celeba', 'utk', 'fair_face'])
    parser.add_argument('--model_path_version', type=str, help='Version of the model path', default="v1")
    parser.add_argument('--baseline_model_path', type=str, help='Version of the model path')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=512)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device for training (cpu/gpu/mps)')

    args = parser.parse_args()

    main(args)

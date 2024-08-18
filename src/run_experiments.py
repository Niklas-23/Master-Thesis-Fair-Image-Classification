import argparse
import os
from datetime import datetime

import torch
from torchvision import transforms

from src.benchmark_models.ResNet18 import ResNet18
from src.datasets.CelebA_dataset import CelebADataset
from src.datasets.FairFaceDataset import FairFaceDataset
from src.datasets.LFWDataset import LFWDataset
from src.datasets.UTKFaceDataset import UTKFaceDataset
from src.fairness_libraries.entangling_disentangling_bias.EnDModel import ResNet18EnD
from src.fairness_libraries.fair_contrastive_learning.FairSupConModels import FairSupConResNet18, LinearClassifier
from src.fairness_libraries.fcro.FCRO_model import ResNet18FCRO
from src.fairness_libraries.shared_adversarial_encoder.SharedEncoderResNet18 import SharedAdversarialEncoderResNet18
from src.mitigation.AdversarialDebiasing import AdversarialDebiasingBenchmark
from src.mitigation.BaseLossOptimization import BaseLossOptimizationBenchmark
from src.mitigation.BlackboxMulticlassBalancer import BlackboxMulticlassBalancerBechmark
from src.mitigation.DomainIndependentTraining import DomainIndependentTrainingBenchmark
from src.mitigation.EntanglingDisentangling import EntanglingDisentanglingBenchmark
from src.mitigation.FairContrastiveLearning import TwoCropTransform, FairContrastiveLearningBenchmark
from src.mitigation.FairFeatureDistillation import FairFeatureDistillationBenchmark
from src.mitigation.ImageSketching import ImageSketchingBenchmark
from src.mitigation.OrthogonalRepresentations import OrthogonalRepresentationsBenchmark
from src.mitigation.SharedAdversarialEncoder import SharedAdversarialEncoderBenchmark


def main(args):
    print("Start experiments")
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
        train_transform_utk = transforms.Compose([
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000))
        ])
        experiment_dataset = UTKFaceDataset(transform=train_transform_utk)
        model_path = f"utk_face/{model_path_version}"
        baseline_model_path = baseline_model_path + "utk_face" + baseline_model_version
    elif dataset_name == "fair_face":
        experiment_dataset = FairFaceDataset(target_name="gender")
        model_path = f"fair_face/{model_path_version}"
        baseline_model_path = baseline_model_path + "fair_face" + baseline_model_version
    elif dataset_name == "lfw":
        #experiment_dataset = LFWDataset(target_name="Strong Nose-Mouth Lines", csv_name="lfw_dataset_male__strong_nose_mouth_skewed_0.8")
        experiment_dataset = LFWDataset(target_name="Wearing Lipstick")
        model_path = f"lfw/{model_path_version}"
        baseline_model_path = baseline_model_path + "lfw" + baseline_model_version

    mitigation_approaches = []
    mitigation_approaches.append(get_image_sketching_benchmark(experiment_dataset, device, batch_size, model_path))
    mitigation_approaches.append(get_adversarial_debiasing_benchmark(experiment_dataset, device, batch_size, model_path))
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
    mitigation_approaches.append(get_fair_contrastive_learning_benchmark(experiment_dataset, device, batch_size, model_path))

    for approach in mitigation_approaches:
        approach.run_benchmark()


def get_adversarial_debiasing_benchmark(dataset, device, batch_size, model_path):
    resnet18_model = ResNet18(num_classes=dataset.get_num_classes_target())
    return AdversarialDebiasingBenchmark(resnet18_model, dataset, device=device,
                                         model_path=f"adversarial_debiasing/{model_path}",
                                         batch_size=batch_size)


def get_base_loss_optimization_benchmark(dataset, device, batch_size, model_path):
    resnet18_model = ResNet18(num_classes=dataset.get_num_classes_target())
    return BaseLossOptimizationBenchmark(resnet18_model, dataset,
                                         device=device,
                                         model_path=f"base_loss_optimization/{model_path}",
                                         batch_size=batch_size)


def get_blackbox_multiclass_balancer_benchmark(dataset, device, batch_size, model_path, baseline_model_path):
    resnet18_model = ResNet18(num_classes=dataset.get_num_classes_target())
    return BlackboxMulticlassBalancerBechmark(resnet18_model, dataset, device=device,
                                              model_path=f"blackbox_balancer/{model_path}", batch_size=batch_size,
                                              baseline_model_path=baseline_model_path)


def get_domain_independent_training_benchmark(dataset, device, batch_size, model_path):
    model_size = dataset.get_num_classes_target() * dataset.get_num_classes_protected_feature()
    resnet18_model = ResNet18(model_size)
    return DomainIndependentTrainingBenchmark(resnet18_model, dataset, device=device,
                                              model_path=f"domain_independent_learning/{model_path}",
                                              batch_size=batch_size)


def get_entangling_disentangling_benchmark(dataset, device, batch_size, model_path):
    resnet18_model = ResNet18EnD(num_classes=dataset.get_num_classes_target())
    return EntanglingDisentanglingBenchmark(resnet18_model, dataset, device=device,
                                            model_path=f"entangling_disentangling/{model_path}",
                                            batch_size=batch_size)


def get_fair_contrastive_learning_benchmark(dataset, device, batch_size, model_path):
    dataset_class = type(dataset)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(128,128), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000))
    ])

    target_name, protected_feature_name = dataset.get_target_protected_feature_names()
    contrastive_dataset = dataset_class(input_shape=(128, 128), transform=TwoCropTransform(train_transform), target_name=target_name, protected_feature_name=protected_feature_name)
    classifier_dataset = dataset

    contrastive_resnet18_model = FairSupConResNet18()
    classifier_model = LinearClassifier(num_classes=classifier_dataset.get_num_classes_target())

    return FairContrastiveLearningBenchmark(contrastive_resnet18_model, classifier_model,
                                            contrastive_dataset, classifier_dataset,
                                            device=device,
                                            model_path=f"fair_contrastive/{model_path}",
                                            batch_size=batch_size)


def get_fair_feature_distillation_benchmark(dataset, device, batch_size, model_path, baseline_model_path):
    state = torch.load(baseline_model_path)
    student_model = ResNet18(num_classes=dataset.get_num_classes_target())
    teacher_model = ResNet18(num_classes=dataset.get_num_classes_target())
    teacher_model.load_state_dict(state['model'])

    return FairFeatureDistillationBenchmark(student_model, teacher_model, dataset, device=device,
                                            model_path=f"fair_feat_distill/{model_path}",
                                            batch_size=batch_size)


def get_image_sketching_benchmark(dataset, device, batch_size, model_path):
    resnet18_model = ResNet18(num_classes=dataset.get_num_classes_target())
    return ImageSketchingBenchmark(resnet18_model, dataset, device=device,
                                   model_path=f"image_sketching/{model_path}",
                                   batch_size=batch_size)


def get_orthogonal_representations_benchmark(dataset, device, batch_size, model_path):
    model_t = ResNet18FCRO(num_classes=dataset.get_num_classes_target())
    model_a = ResNet18FCRO(num_classes=dataset.get_num_classes_protected_feature())
    return OrthogonalRepresentationsBenchmark(model_t, model_a, dataset, device=device,
                                              model_path=f"orthogonal_representations/{model_path}",
                                              batch_size=batch_size)


def get_shared_adv_encoder_benchmark(dataset, device, batch_size, model_path):
    resnet18_model = SharedAdversarialEncoderResNet18(num_classes_target=dataset.get_num_classes_target(), num_classes_protected_features=dataset.get_num_classes_protected_feature())

    return SharedAdversarialEncoderBenchmark(resnet18_model, dataset, device=device,
                                             model_path=f"shared_adv_encoder/{model_path}",
                                             batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified parameters')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', choices=['lfw', 'celeba', 'utk', 'fair_face'])
    parser.add_argument('--model_path_version', type=str, help='Version of the model path', default="v1")
    parser.add_argument('--baseline_model_path', type=str, help='Version of the model path')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=512)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', ',mps'],
                        help='Device for training (cpu/gpu/mps)')

    args = parser.parse_args()

    main(args)

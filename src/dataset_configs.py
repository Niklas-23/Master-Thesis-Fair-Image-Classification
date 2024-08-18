import os

from src.constants import CELEBA_DATASET_DIRECTORY, CELEBA_IMAGE_DIRECTORY, CELEBA_DATASET, FAIR_FACE_DATASET, \
    FAIR_FACE_IMAGE_DIRECTORY, FAIR_FACE_DATASET_DIRECTORY, UTK_FACE_DATASET, LFW_DATASET, UTK_FACE_DATASET_DIRECTORY, \
    UTK_FACE_IMAGE_DIRECTORY, LFW_DATASET_DIRECTORY, LFW_IMAGE_DIRECTORY


def celeba_config():
    return CELEBA_DATASET_DIRECTORY, CELEBA_IMAGE_DIRECTORY


def fair_face_config():
    return FAIR_FACE_DATASET_DIRECTORY, FAIR_FACE_IMAGE_DIRECTORY


def utk_face_config():
    return UTK_FACE_DATASET_DIRECTORY, UTK_FACE_IMAGE_DIRECTORY


def lfw_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lfw_dataset_image_root_dir = os.path.join(current_dir, f"../datasets_data/{LFW_DATASET_DIRECTORY}/{LFW_IMAGE_DIRECTORY}")
    directory_names = []
    for item in os.listdir(lfw_dataset_image_root_dir):
        if os.path.isdir(os.path.join(lfw_dataset_image_root_dir, item)):
            directory_names.append(f"{LFW_IMAGE_DIRECTORY}/{item}")

    print(directory_names)
    return LFW_DATASET_DIRECTORY, directory_names


switch_cases = {
    CELEBA_DATASET: celeba_config,
    FAIR_FACE_DATASET: fair_face_config,
    UTK_FACE_DATASET: utk_face_config,
    LFW_DATASET: lfw_config
}


def get_dataset_config(case):
    return switch_cases.get(case, lambda: "Invalid dataset name")()

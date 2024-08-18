from typing import Union

from src.constants import SKETCH_IMAGE_DIRECTORY
from src.fairness_libraries.informative_drawings.informative_drawings_model import InformativeDrawingsInference
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

inference_predictor = InformativeDrawingsInference()


def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def sketch_existence(sketch_image_path: str):
    return os.path.exists(sketch_image_path)


def create_sketch_images(dataset_directory: str, image_directory: Union[str | list[str]], max_workers: int = 3):

    current_src_path = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.join(current_src_path, '../../')

    if isinstance(image_directory, str):
        image_directory = f"{rootdir}/datasets_data/{dataset_directory}/{image_directory}"
        sketch_directory = f"{rootdir}/datasets_data/{dataset_directory}/{SKETCH_IMAGE_DIRECTORY}"

        run_sketch_image_creation(sketch_directory, image_directory, max_workers)
    else:
        for path in image_directory:
            print(f"Selected image directory path: {path}")
            image_directory = f"{rootdir}/datasets_data/{dataset_directory}/{path}"
            sketch_directory = f"{rootdir}/datasets_data/{dataset_directory}/{SKETCH_IMAGE_DIRECTORY}/{path}"

            run_sketch_image_creation(sketch_directory, image_directory, max_workers)


def run_sketch_image_creation(sketch_directory:str, image_directory:str, max_workers:int):

    if os.path.exists(sketch_directory) and len(os.listdir(sketch_directory)) >= len(os.listdir(image_directory)):
        print("Sketch images already exist")
    else:
        create_directory(sketch_directory)
        image_names = os.listdir(image_directory)

        def process_image(image_name):
            image_path = os.path.join(image_directory, image_name)
            save_path = os.path.join(sketch_directory, image_name)

            if not sketch_existence(save_path):
                sketch_image = inference_predictor.predict(image_path)
                sketch_image.save(save_path)

        print("Starting to create sketch images")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_image, image_names), total=len(image_names)))

        print("Sketch images created successfully")

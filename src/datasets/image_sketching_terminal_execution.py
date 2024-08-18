from src.dataset_configs import get_dataset_config
from src.util.image_sketching import create_sketch_images


def main():
    print("Sketch images generation")
    dataset_name = input("Select dataset to create sktech images for. Possible options: celeba. Your choice: ")
    dataset_directory, image_directory = get_dataset_config(dataset_name)
    max_workers = input("Max workers setting: ")
    create_sketch_images(dataset_directory, image_directory, int(max_workers))


if __name__ == "__main__":
    main()

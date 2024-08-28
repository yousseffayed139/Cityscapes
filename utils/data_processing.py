import os
import pandas as pd

def collect_image_paths(base_dir):
    """
    Collect image and mask paths from a base directory with 'images' and 'gtFine' subdirectories.
    
    :param base_dir: Base directory containing 'images' and 'gtFine' subdirectories.
    :return: List of tuples containing image and mask paths.
    """
    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'gtFine')
    
    image_paths = []
    mask_paths = []

    for city in os.listdir(image_dir):
        city_image_dir = os.path.join(image_dir, city)
        city_mask_dir = os.path.join(mask_dir, city)

        for img_file in os.listdir(city_image_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(city_image_dir, img_file)
                mask_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                mask_path = os.path.join(city_mask_dir, mask_file)

                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

    # Save paths to a CSV file
    data = {'image_path': image_paths, 'mask_path': mask_paths}
    df = pd.DataFrame(data)
    df.to_csv('dataset_paths.csv', index=False)

    return image_paths, mask_paths

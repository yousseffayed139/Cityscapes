import os
import torch
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_processing import collect_image_paths
from utils.transforms import Resize, RandomRotation, ToTensor, NormalizeImage, NormalizeLabels
from utils.metrics import iou_metric
from models.unet_attention import UNetWithAttention
from PIL import Image
from matplotlib import cm
import numpy as np


def apply_color_map(image, cmap='tab20', num_classes=21):
    colormap = cm.get_cmap(cmap, num_classes)  # Correct method to retrieve colormap
    color_mapped_image = colormap(image / (num_classes - 1))[:, :, :3]
    color_mapped_image = (color_mapped_image * 255).astype(np.uint8)
    return color_mapped_image


def save_combined_image(pred_image, gt_image, output_path, padding=10, title_height=30):
    pred_image_pil = Image.fromarray(pred_image)
    gt_image_pil = Image.fromarray(gt_image)

    width, height = pred_image_pil.size

    # Create a new image with extra padding and space for titles
    combined_width = 2 * width + padding
    combined_height = height + title_height

    combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    # Draw titles
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    draw.text((width // 2, 5), "Ground Truth", font=font, fill=(0, 0, 0))
    draw.text((width + padding + width // 2, 5), "Predicted", font=font, fill=(0, 0, 0))

    # Paste images with padding
    combined_image.paste(gt_image_pil, (0, title_height))
    combined_image.paste(pred_image_pil, (width + padding, title_height))

    combined_image.save(output_path)




def load_data(csv_file, transform=None):
    """
    Load dataset for testing using a CSV file with paths.
    
    :param csv_file: Path to the CSV file containing image and mask paths.
    :param transform: Transformations to apply.
    :return: DataLoader instance.
    """
    df = pd.read_csv(csv_file)
    image_paths = df['image_path'].tolist()
    mask_paths = df['mask_path'].tolist()

    class CityscapesDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, mask_paths, transform=None):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            mask = Image.open(self.mask_paths[idx])

            sample = {'image': image, 'mask': mask}

            if self.transform:
                sample = self.transform(sample)

            return sample['image'], sample['mask']

    return CityscapesDataset(image_paths, mask_paths, transform)

def main():
    
    # Get dataset and model paths from environment variables
    base_dir = os.getenv('BASE_DIR', '/data/base_dir')
    model_weights_path = os.getenv('MODEL_WEIGHTS_PATH', '/data/model_weights.pth')
    # model_weights_path="/home/youssef/Projects/CityScape/Mount/model_weights.pth"
    # Collect image and mask paths and save to CSV
    collect_image_paths(base_dir)
    
    # Define transformations
    transform = transforms.Compose([
        Resize((256, 256)),
        ToTensor(),
        NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        NormalizeLabels()
    ])

    # Load data from CSV
    dataset = load_data('dataset_paths.csv', transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetWithAttention(in_channels=3, out_channels=21).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    model.eval()
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)

            iou = iou_metric(outputs, masks)
            print(f'Batch {i} IoU: {iou:.4f}')
            total_iou += iou
            num_batches += 1

            # Save predictions
            for j in range(images.size(0)):
                pred_image = outputs[j].cpu().numpy()
                gt_image = masks[j].cpu().numpy()

                # Apply color mapping
                pred_image_colored = apply_color_map(pred_image)
                gt_image_colored = apply_color_map(gt_image)

                # Save combined image
                save_combined_image(pred_image_colored, gt_image_colored, 
                                    f'/Output/combined_{i}_{j}.png')

    # Calculate and print mean IoU over all batches
    mean_iou = total_iou / max(num_batches, 1)
    print(f'Mean IoU over all batches: {mean_iou:.4f}')

if __name__ == "__main__":
    main()

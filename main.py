import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_processing import collect_image_paths
from utils.transforms import Resize, RandomRotation, ToTensor, NormalizeImage, NormalizeLabels
from utils.metrics import iou_metric
from models.unet_attention import UNetWithAttention
from PIL import Image
import numpy as np

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

    # Collect image and mask paths and save to CSV
    collect_image_paths(base_dir)
    
    # Define transformations
    transform = transforms.Compose([
        Resize((256, 256)),
        RandomRotation(10),
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
    model.load_state_dict(torch.load(model_weights_path))

    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)

            iou = iou_metric(outputs, masks)
            print(f'Batch {i} IoU: {iou:.4f}')

            # Save predictions
            for j in range(images.size(0)):
                pred_image = outputs[j].cpu().numpy()
                pred_image = Image.fromarray(pred_image.astype(np.uint8))
                pred_image.save(f'/output/pred_{i}_{j}.png')

                gt_image = masks[j].cpu().numpy()
                gt_image = Image.fromarray(gt_image.astype(np.uint8))
                gt_image.save(f'/output/gt_{i}_{j}.png')

if __name__ == "__main__":
    main()

import torch

def compute_accuracy(outputs, masks):
    outputs = torch.argmax(outputs.float(), dim=1)
    correct = (outputs == masks).float().sum()
    total = torch.numel(masks)
    accuracy = correct / total
    return accuracy.item()

import torch

def iou_metric(outputs, targets, num_classes=21):
    """
    Compute Intersection over Union (IoU) for each class, excluding class ID 0.
    
    Args:
    - outputs (torch.Tensor): Model predictions of shape [batch_size, num_classes, height, width].
    - targets (torch.Tensor): Ground truth labels of shape [batch_size, height, width].
    - num_classes (int): Number of classes including background (0 to 20).
    
    Returns:
    - mean_iou (float): Average IoU across all classes except class ID 0.
    """
    print("Before softmax and argmax:", outputs.size())
    
    # Convert outputs to class predictions
    # outputs = torch.softmax(outputs.float(), dim=1).argmax(dim=1)  # Shape: [batch_size, height, width]
    
    print("After softmax and argmax:", outputs.size())
    
    iou_per_class = torch.zeros(num_classes, dtype=torch.float32)
    num_valid_classes = 0
    
    for cls in range(1, num_classes):  # Start from 1 to ignore class ID 0
        pred = (outputs == cls).float()
        target = (targets == cls).float()
        
        intersection = (pred * target).sum().item()
        union = (pred + target).sum().item() - intersection
        
        if union > 0:  # Only calculate IoU if there is at least one pixel in the union
            iou_per_class[cls] = intersection / (union + 1e-6)  # Add small value to avoid division by zero
            num_valid_classes += 1
    
    # Compute mean IoU for classes that are present in the target masks
    if num_valid_classes > 0:
        mean_iou = iou_per_class[1:num_classes].sum() / num_valid_classes
    else:
        mean_iou = 0.0
    
    return mean_iou


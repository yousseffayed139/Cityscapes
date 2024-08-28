import torch

def compute_accuracy(outputs, masks):
    outputs = torch.argmax(outputs, dim=1)
    correct = (outputs == masks).float().sum()
    total = torch.numel(masks)
    accuracy = correct / total
    return accuracy.item()

def iou_metric(outputs, targets, num_classes=21):
    outputs = torch.softmax(outputs, dim=1).argmax(dim=1)
    iou_per_class = torch.zeros(num_classes, dtype=torch.float32)
    num_valid_classes = 0
    
    for cls in range(1, num_classes):
        pred = (outputs == cls).float()
        target = (targets == cls).float()
        intersection = (pred * target).sum().item()
        union = (pred + target).sum().item() - intersection
        
        if union > 0:
            iou_per_class[cls] = intersection / (union + 1e-6)
            num_valid_classes += 1
    
    mean_iou = iou_per_class[1:num_classes].sum() / max(num_valid_classes, 1)
    return mean_iou

import torch
from tqdm import tqdm


def validate(model, val_loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The VQA model.
        val_loader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (torch.device): Device to run computations on.

    Returns:
        float: Average validation loss.
        float: Validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, questions, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

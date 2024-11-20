import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.training.logger import Logger
from src.data.dataset import NIHChestXrayDataset
from src.data.transforms import get_transforms
from src.models.vqa_model import VQAModel

def train_one_step(model, optimizer, criterion, images, questions, labels, device):
    """
    Performs one training step, including forward and backward passes.

    Args:
        model (torch.nn.Module): The VQA model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function to compute training loss.
        images (torch.Tensor): Batch of input images.
        questions (list[str]): Batch of corresponding questions.
        labels (torch.Tensor): Ground truth labels for the batch.
        device (torch.device): Device to run computations on (CPU/GPU).

    Returns:
        float: Loss for the current training step.
    """
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images, questions)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The VQA model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function to compute validation loss.
        device (torch.device): Device to run computations on (CPU/GPU).

    Returns:
        tuple:
            float: Average validation loss over the dataset.
            float: Validation accuracy (correct predictions / total samples).
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


def train():
    """
    Trains the VQA model over multiple epochs, with periodic validation.

    This function:
    - Loads and preprocesses the training and validation datasets.
    - Initializes the model, optimizer, and loss function.
    - Performs training for the specified number of epochs.
    - Validates the model at the end of each epoch.
    - Saves the model checkpoints after every epoch.

    Returns:
        None. Logs training and validation metrics and saves checkpoints.
    """
    experiment_name = "vgg16_bio_clinical_bert_concat"
    logger = Logger(log_dir="logs", experiment_name=experiment_name)

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    IMAGE_DIM = 1024
    TEXT_DIM = 512
    FUSED_DIM = 1024
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to datasets
    train_csv = "data/processed/train_vqa_dataset.csv"
    test_csv = "data/processed/test_vqa_dataset.csv"

    # Load transforms
    transforms = get_transforms(image_size=(1024, 1024))

    # Datasets and DataLoaders
    train_dataset = NIHChestXrayDataset(
        csv_path=train_csv,
        transform=transforms["train"],
        mode="train"
    )
    val_dataset = NIHChestXrayDataset(
        csv_path=test_csv,
        transform=transforms["val"],
        mode="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = VQAModel(
        image_dim=IMAGE_DIM,
        text_dim=TEXT_DIM,
        fused_dim=FUSED_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, questions, labels in tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100):
            # One training step
            loss = train_one_step(model, optimizer, criterion, images, questions, labels, DEVICE)
            running_loss += loss
            
        logger.log(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # Log training loss
        avg_train_loss = running_loss / len(train_loader)
        logger.log(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}")

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, DEVICE)
        logger.log(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/vqa_model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()

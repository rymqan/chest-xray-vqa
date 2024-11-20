import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from src.models import VQAModel
from src.data.dataset import NIHChestXrayDataset
from src.data.transforms import get_transforms
from src.training.train_utils import train_one_step
from src.training.val_utils import validate

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_DIM = 1024
TEXT_DIM = 512
FUSED_DIM = 1024
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Load transforms
    transforms = get_transforms(image_size=(224, 224))

    # Datasets and DataLoaders
    train_dataset = NIHChestXrayDataset(
        data_path="data/processed/",
        transform=transforms["train"],
        mode="train"
    )
    val_dataset = NIHChestXrayDataset(
        data_path="data/processed/",
        transform=transforms["val"],
        mode="val"
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
        for images, questions, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # One training step
            loss = train_one_step(model, optimizer, criterion, images, questions, labels, DEVICE)
            running_loss += loss

        # Log training loss
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}")

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save model checkpoint
        os.makedirs("experiments/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"experiments/checkpoints/vqa_model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()

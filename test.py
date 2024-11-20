import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src.models import VQAModel
from src.data.dataset import NIHChestXrayDataset
from src.data.transforms import get_transforms

# Hyperparameters and Paths
BATCH_SIZE = 16
IMAGE_DIM = 1024
TEXT_DIM = 512
FUSED_DIM = 1024
NUM_CLASSES = 2
CHECKPOINT_PATH = "experiments/checkpoints/vqa_model_epoch_10.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_predictions(model, test_loader, class_names, device, num_samples=5):
    """
    Visualizes predictions for a few test samples.

    Args:
        model (torch.nn.Module): The trained VQA model.
        test_loader (DataLoader): DataLoader for the test set.
        class_names (list[str]): List of class labels corresponding to indices.
        device (torch.device): Device to run computations on (CPU/GPU).
        num_samples (int, optional): Number of samples to visualize. Default is 5.

    Returns:
        None. Displays the images, predicted labels, and true labels.
    """
    model.eval()
    count = 0

    with torch.no_grad():
        for images, questions, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, questions)
            _, predicted = torch.max(outputs, dim=1)

            for i in range(images.size(0)):
                if count >= num_samples:
                    return
                
                # Convert tensor to image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * 0.229) + 0.485  # Undo normalization (mean and std)
                img = img.clip(0, 1)

                plt.imshow(img)
                plt.title(f"Question: {questions[i]}\nPredicted: {class_names[predicted[i].item()]}\nTrue: {class_names[labels[i].item()]}")
                plt.axis("off")
                plt.show()

                count += 1

# Function to test the model
def test():
    """
    Tests the trained VQA model on the test set.

    This function:
    - Loads the test dataset and DataLoader.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set.
    - Computes and displays metrics (loss, accuracy, confusion matrix).
    - Visualizes a few sample predictions.

    Returns:
        None. Prints metrics and displays visualizations.
    """
    # Load transforms
    transforms = get_transforms(image_size=(224, 224))

    # Test Dataset and DataLoader
    test_dataset = NIHChestXrayDataset(
        data_path="data/processed/",
        transform=transforms["val"],
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = ["No", "Yes"]

    # Load model
    model = VQAModel(
        image_dim=IMAGE_DIM,
        text_dim=TEXT_DIM,
        fused_dim=FUSED_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Metrics
    total_samples = 0
    correct_predictions = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, questions, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Store true and predicted labels for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute final metrics
    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Visualize predictions
    visualize_predictions(model, test_loader, class_names, DEVICE)


if __name__ == "__main__":
    test()

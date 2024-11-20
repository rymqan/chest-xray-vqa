import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class NIHChestXrayDataset(Dataset):
    def __init__(self, data_path, transform=None, mode="train"):
        """
        NIH Chest X-ray dataset loader.

        Args:
            data_path (str): Path to the processed data directory.
            transform (callable, optional): Image transform pipeline.
            mode (str): "train", "val", or "test".
        """
        self.data_path = data_path
        self.transform = transform
        self.mode = mode

        # Load the CSV file
        metadata_path = f"{data_path}/metadata/Data_Entry_2017.csv"
        self.metadata = pd.read_csv(metadata_path)

        # Filter train/val/test split
        self.metadata = self.metadata[self.metadata["Dataset"] == mode]

        # Prepare image paths and labels
        self.image_paths = self.metadata["Image Index"].values
        self.labels = self.metadata["Finding Labels"].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = f"{self.data_path}/images/{self.image_paths[idx]}"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load corresponding label (example: classification target)
        label = torch.tensor(self.labels[idx])  # Adjust for your task
        
        # Example question generation (placeholder)
        question = "What is the diagnosis?"  # Customize for VQA

        return image, question, label

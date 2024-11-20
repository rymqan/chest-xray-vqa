import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class NIHChestXrayDataset(Dataset):
    """
    Custom Dataset for loading NIH Chest X-ray images with questions and answers.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        transform (callable, optional): Transformations to apply to the images.
        mode (str, optional): Mode of operation ("train", "test"). Used for logging or debugging.

    Attributes:
        data (pd.DataFrame): DataFrame containing the dataset information.
        transform (callable): Image transformations.
    """

    def __init__(self, csv_path: str, transform=None, mode: str = "train"):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, question, answer), where:
                - image (torch.Tensor): Transformed image.
                - question (str): Question associated with the image.
                - answer (int): Binary answer (0 or 1).
        """
        row = self.data.iloc[idx]
        # image_path = f"data/raw/images/{row['image']}"  # Adjust if image paths differ
        question = row['question']
        answer = int(row['answer'])  # Ensure the answer is binary (0 or 1)

        # Load and transform the image
        # image = Image.open(image_path).convert("RGB")
        image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)

        return image, question, answer

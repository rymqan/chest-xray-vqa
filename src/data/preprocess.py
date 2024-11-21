import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Define paths
RAW_DATA_PATH = "data/raw/augmented_nih_vqa_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/preprocessed_vqa_dataset.csv"
TRAIN_DATA_PATH = "data/processed/train_vqa_dataset.csv"
TEST_DATA_PATH = "data/processed/test_vqa_dataset.csv"


def preprocess_vqa_dataset(input_csv: str, output_csv: str, random_state: int = 123456):
    """
    Preprocesses the VQA dataset by shuffling and ensuring proper binary formatting of answers.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the processed CSV file.
        random_state (int, optional): Random state for reproducibility. Default is 123456.

    Returns:
        None: Saves the processed dataset to the specified output path.
    """
    # Load the dataset
    data = pd.read_csv(input_csv)

    # Ensure the answer column is binary
    data['answer'] = data['answer'].astype(bool)

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Save the processed dataset
    data.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to {output_csv}")


def split_dataset(input_csv: str, train_csv: str, test_csv: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the preprocessed dataset into training and testing subsets.

    Args:
        input_csv (str): Path to the preprocessed dataset.
        train_csv (str): Path to save the training subset.
        test_csv (str): Path to save the testing subset.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
        None: Saves the train and test datasets to their respective paths.
    """
    # Load the preprocessed dataset
    data = pd.read_csv(input_csv)

    # Perform train-test split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # Save the splits
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)
    print(f"Training set saved to {train_csv}")
    print(f"Test set saved to {test_csv}")


if __name__ == "__main__":
    # Create processed data directory if it doesn't exist
    os.makedirs("data/processed/", exist_ok=True)

    # Step 1: Preprocess the dataset
    print("Preprocessing the raw dataset...")
    preprocess_vqa_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    # Step 2: Split the dataset into train and test
    print("Splitting the preprocessed dataset into train and test subsets...")
    split_dataset(PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH)

    print("Preprocessing completed successfully!")

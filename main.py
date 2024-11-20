import torch

from src.models.vqa_model import VQAModel


def test_pipeline():
    # Parameters
    batch_size = 4
    num_classes = 100
    image_dim = 1024
    text_dim = 512
    fused_dim = 1024
    
    # Instantiate the model
    vqa_model = VQAModel(image_dim=image_dim, text_dim=text_dim, fused_dim=fused_dim, num_classes=num_classes)
    
    # Generate mock data
    images = torch.randn(batch_size, 3, 224, 224)  # Batch of RGB images (preprocessed dimensions)
    questions = [
        "What is the diagnosis?",
        "Are there signs of pneumonia?",
        "Is there evidence of fractures?",
        "What is the heart size?"
    ]  # Mock questions
    
    # Run the forward pass
    predictions = vqa_model(images, questions)  # Output should be [batch_size, num_classes]
    
    # Print dimensions at each stage (debugging purposes)
    print("Test VQA Pipeline:")
    print(f"Input Images Shape: {images.shape} (Expected: [batch_size, 3, 224, 224])")
    print(f"Questions: {questions} (Expected: List of strings, batch_size elements)")
    print(f"Output Predictions Shape: {predictions.shape} (Expected: [batch_size, {num_classes}])")
    
    # Check dimensions
    assert predictions.shape == (batch_size, num_classes), "Output dimensions do not match expectations"
    print("Pipeline test passed! All dimensions are correct.")

# Run the test
if __name__ == "__main__":
    test_pipeline()

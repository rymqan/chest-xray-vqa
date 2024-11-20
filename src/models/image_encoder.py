import torch
import torch.nn as nn
from torchvision import models

class VGG16Encoder(nn.Module):
    """
    VGG-16-based image encoder for feature extraction.

    The model uses the convolutional layers of the VGG-16 model and
    replaces the classifier to output a custom feature dimension.

    Attributes:
        features (nn.Sequential): Convolutional layers from VGG-16.
        avgpool (nn.AdaptiveAvgPool2d): Average pooling layer to reduce spatial dimensions.
        fc (nn.Sequential): Custom fully connected layers for feature extraction.
    """
    def __init__(self, output_dim=1024):
        """
        Initialize the VGG-16 encoder.

        Args:
            output_dim (int): The dimensionality of the output feature vector.
        """
        super(VGG16Encoder, self).__init__()
        # Load the VGG-16 model
        vgg16 = models.vgg16(weights=None)
        
        # Extract the convolutional and pooling layers (feature extractor)
        self.features = vgg16.features
        
        # Retain the original average pooling layer
        self.avgpool = vgg16.avgpool

        # Replace the original classifier with a custom one
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the VGG-16 encoder.

        Args:
            x (torch.Tensor): Batch of images with shape [batch_size, 3, 224, 224].

        Returns:
            torch.Tensor: Feature vectors with shape [batch_size, output_dim].
        """
        # Pass input through convolutional layers
        x = self.features(x)
        
        # Apply average pooling to reduce spatial dimensions
        x = self.avgpool(x)
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Pass through the fully connected layers
        x = self.fc(x)
        return x

from src.models.image_encoder import VGG16Encoder
from src.models.text_encoder import BioClinicalBERTEncoder
from src.models.fusion import FusionModule
from src.models.decoder import Decoder

import torch.nn as nn


class VQAModel(nn.Module):
    def __init__(self, image_dim, text_dim, fused_dim, num_classes):
        """
        Visual Question Answering (VQA) model combining image encoder, text encoder, fusion module, and decoder.

        Args:
            image_dim (int): Dimensionality of image features.
            text_dim (int): Dimensionality of text features.
            fused_dim (int): Dimensionality of fused features.
            num_classes (int): Number of possible answers.
        """
        super(VQAModel, self).__init__()
        self.image_encoder = VGG16Encoder()  # Use the VGG16Encoder class
        self.text_encoder = BioClinicalBERTEncoder(output_dim=text_dim)  # Use the BioClinicalBERTEncoder class
        self.fusion = FusionModule(image_dim=image_dim, text_dim=text_dim, fused_dim=fused_dim)
        self.decoder = Decoder(fused_dim=fused_dim, num_classes=num_classes)

    def forward(self, images, questions):
        """
        Forward pass for the VQA model.

        Args:
            images (torch.Tensor): Batch of images (shape: [batch_size, 3, H, W]).
            questions (list[str]): Batch of questions (as strings).

        Returns:
            torch.Tensor: Predicted probabilities for each class (shape: [batch_size, num_classes]).
        """
        # Encode image and text
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(questions)
        
        # Fuse features
        fused_features = self.fusion(image_features, text_features)
        
        # Decode to generate answer
        predictions = self.decoder(fused_features)
        return predictions

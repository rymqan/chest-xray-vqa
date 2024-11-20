def train_one_step(model, optimizer, criterion, images, questions, labels, device):
    """
    Performs one training step.

    Args:
        model (nn.Module): The VQA model.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        images (torch.Tensor): Batch of images.
        questions (list): Batch of questions.
        labels (torch.Tensor): Batch of labels.
        device (torch.device): Device to run computations on.

    Returns:
        float: Loss for the current batch.
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

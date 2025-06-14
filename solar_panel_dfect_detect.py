# Brings in PyTorch, the main library for doing deep‑learning stuff in Python
# import torch
import torch

# Imports the neural network module containing layers and loss functions
# import torch.nn as nn
import torch.nn as nn

# Imports optimization algorithms like AdamW under the alias optim
# import torch.optim as optim
import torch.optim as optim

# Datasets helps load image folders; transforms lets us preprocess images
# from torchvision import datasets, transforms
from torchvision import datasets, transforms

# DataLoader wraps a dataset to provide batching, shuffling, etc.
# from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

# ViT model for classification and feature extractor for preprocessing
# from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Check if a GPU with CUDA is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Group image preprocessing steps into one pipeline
transform = transforms.Compose([
    # Resize images to 224x224 pixels, as ViT expects
    transforms.Resize((224, 224)),
    # Randomly flip images left-right to augment data
    transforms.RandomHorizontalFlip(),
    # Randomly flip images up-down to augment data
    transforms.RandomVerticalFlip(),
    # Convert images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize images using ImageNet statistics
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the directory containing subfolders for each class
data_dir = r'C:\Users\Aanya\Downloads\Training_data\Faulty_solar_panel'

# Load images from folders, applying our transform for both training and validation
t rain_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
val_dataset   = datasets.ImageFolder(root=data_dir, transform=transform)

# Automatically count how many classes (subfolders) are present
num_classes = len(train_dataset.classes)
print(f'Number of classes: {num_classes}')

# Wrap datasets in DataLoaders to handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

# Load a pre-trained Vision Transformer, adapting it to our number of classes
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes  # Set final layer to match our classes
)
# Move the model to GPU or CPU\ nmodel.to(device)

# Cross entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()
# AdamW optimizer with a small learning rate for fine-tuning
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training loop function to train and validate over multiple epochs
def train(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        # Set model to training mode (enables dropout, gradients, etc.)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate over each batch of training data
        for inputs, labels in train_loader:
            # Move data to the correct device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero out gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: compute model outputs
            outputs = model(inputs).logits
            # Compute loss comparing outputs and true labels
            loss = criterion(outputs, labels)
            # Backward pass: compute gradients
            loss.backward()
            # Update model weights
            optimizer.step()

            # Accumulate loss for reporting
            running_loss += loss.item()
            # Determine predicted class for each sample
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Print average loss and accuracy for this epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100.*correct/total}%')

        # Run validation after each training epoch
        validate(model, val_loader)

# Validation function to check performance on unseen data
 def validate(model, val_loader):
    # Set model to evaluation mode (disables dropout, no grad tracking)
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs).logits
            # Compute loss
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Print validation loss and accuracy\ n    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100.*correct/total}%')

# Kick off training for specified number of epochs
train(model, train_loader, val_loader, epochs=10)

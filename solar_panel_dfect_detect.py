import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data augmentation and normalization for training/validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),  # Augmentation: random horizontal flip
    transforms.RandomVerticalFlip(),    # Augmentation: random vertical flip
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Path to your training data directory
data_dir = r'C:\Users\Aanya\Downloads\Training_data\Faulty_solar_panel'

# Load datasets (replace with the correct data_dir)
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Automatically detect the number of classes (based on the number of defect folders)
num_classes = len(train_dataset.classes)
print(f'Number of classes: {num_classes}')

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ViT model from Hugging Face (ViT Base Patch16-224)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes  # Use the dynamically detected number of classes
)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
def train(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100.*correct/total}%')

        # Validation phase
        validate(model, val_loader)

# Validation function
def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100.*correct/total}%')

# Start training the model
train(model, train_loader, val_loader, epochs=10)
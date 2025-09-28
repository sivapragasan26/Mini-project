import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = r"C:\Users\somanathan\pulmo-vision\dataset"  # path to your dataset folder
model_save_path = r"C:\Users\somanathan\pulmo-vision\models\VitFinal30_model.pth"
num_epochs = 5
batch_size = 16
learning_rate = 1e-4
num_classes = 4
class_names = ['COVID-19', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral']

# -----------------------------
# Data transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# -----------------------------
# Dataset & Oversampling
# -----------------------------
dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
print(f"Found classes: {dataset.classes}")

# Compute class counts
targets = torch.tensor([y for _, y in dataset])
class_count = torch.tensor([(targets == i).sum().item() for i in range(num_classes)])
print(f"Samples per class: {class_count.tolist()}")

# Create weights for oversampling
class_weights = 1. / class_count
sample_weights = [class_weights[y] for y in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

# -----------------------------
# Model (Vision Transformer)
# -----------------------------
weights = models.ViT_B_16_Weights.DEFAULT
model = models.vit_b_16(weights=weights)
in_features = model.heads.head.in_features
model.heads = nn.Linear(in_features, num_classes)
model = model.to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=running_loss/(total/batch_size), acc=100*correct/total)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# -----------------------------
# Save model
# -----------------------------
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at: {model_save_path}")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuración
ruta_base = '/home/erosales/dl/imgs_avocado'
csv_path = '/home/erosales/dl/dataset_paltas.csv'
batch_size = 32
num_epochs = 15
num_classes = 2  # Índices de 1 a 5

# Dataset personalizado
class AvocadoDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.root_dir, row['File Name'] + '.jpg')
        image = default_loader(image_path)
        label = int(row['Ripening Index Classification']) - 1  # 0-based
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Cargar y dividir datos
df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Ripening Index Classification'], random_state=42)
train_dataset = AvocadoDataset(train_df, ruta_base, transform)
val_dataset = AvocadoDataset(val_df, ruta_base, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Modelo y optimización
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc.requires_grad = True
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Métricas globales
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Entrenamiento
for epoch in range(num_epochs):
    # Entrenamiento
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    # Validación
    model.eval()
    val_loss = 0.0
    correct_val, total_val = 0, 0
    class_loss = [0.0] * num_classes
    class_counts = [0] * num_classes
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_batch = criterion(outputs, labels)
            val_loss += loss_batch.item()

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            # Por muestra
            per_sample_loss = nn.functional.cross_entropy(outputs, labels, reduction='none')
            for i in range(len(labels)):
                lbl = labels[i].item()
                class_loss[lbl] += per_sample_loss[i].item()
                class_counts[lbl] += 1
                total_per_class[lbl] += 1
                if preds[i].item() == lbl:
                    correct_per_class[lbl] += 1

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct_val / total_val)

    print(f"\nÉpoca [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_losses[-1]:.4f} | Acc: {train_accs[-1]*100:.2f}%")
    print(f"Val   Loss: {val_losses[-1]:.4f} | Acc: {val_accs[-1]*100:.2f}%")

    # Mostrar métricas por clase
    print("---- Métricas por clase ----")
    for i in range(num_classes):
        avg_loss = class_loss[i] / class_counts[i] if class_counts[i] > 0 else 0.0
        acc = correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0
        print(f"Clase {i+1}: Val Loss = {avg_loss:.4f}, Accuracy = {acc*100:.2f}%")

# Gráficas finales
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend(); plt.title('Loss por época')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.legend(); plt.title('Accuracy por época')
plt.show()


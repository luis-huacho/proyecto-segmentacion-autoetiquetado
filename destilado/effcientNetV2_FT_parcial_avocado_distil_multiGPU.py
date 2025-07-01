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
from tqdm import tqdm

# Configuración
ruta_base = '/home/erosales/dl/imgs_avocado'
csv_path = '/home/erosales/dl/dataset_paltas.csv'
batch_size = 32
num_epochs = 15
num_classes = 2 

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
        label = int(row['Ripening Index Classification']) - 1
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # EfficientNetV2 necesita resoluciones grandes
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datos
df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Ripening Index Classification'], random_state=42)
train_dataset = AvocadoDataset(train_df, ruta_base, transform)
val_dataset = AvocadoDataset(val_df, ruta_base, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("train_dataset len: ",len(train_dataset),"\n","val_dataset len: ",len(val_dataset))
# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelo: EfficientNetV2 con fine-tuning del bloque "features"
model = models.efficientnet_v2_s(pretrained=True)

# Congelar todas las capas
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo features
for name, param in model.named_parameters():
    if "features" in name:
        param.requires_grad = True

# Reemplazar capa final
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Enviar modelo al dispositivo antes de envolver con DataParallel
model = model.to(device)

# Usar múltiples GPUs si están disponibles
if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Función de pérdida y optimizador
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params_to_update, lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Métricas
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0.0

# Entrenamiento
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    #for images, labels in train_loader:
    for images, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} - Entrenando"):
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

    # Validación con métricas por clase
    model.eval()
    val_loss = 0.0
    correct_val, total_val = 0, 0
    class_loss = [0.0] * num_classes
    class_counts = [0] * num_classes
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        #for images, labels in val_loader:
        for images, labels in tqdm(val_loader, desc=f"Época {epoch+1}/{num_epochs} - Validando"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_batch = criterion(outputs, labels)
            val_loss += loss_batch.item()

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

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

    print("---- Métricas por clase ----")
    for i in range(num_classes):
        avg_loss = class_loss[i] / class_counts[i] if class_counts[i] > 0 else 0.0
        acc = correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0
        print(f"Clase {i+1}: Val Loss = {avg_loss:.4f}, Accuracy = {acc*100:.2f}%")

    # Guardar modelo si mejora val_acc
    current_val_acc = val_accs[-1]
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        nombre_modelo = f"modelo_ep{epoch+1:02d}_acc{current_val_acc*100:.2f}.pt"
        torch.save(model.state_dict(), nombre_modelo)
        print(f"✅ Modelo guardado: {nombre_modelo}")


# Visualización
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss por época'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracy por época'); plt.legend()
plt.show()


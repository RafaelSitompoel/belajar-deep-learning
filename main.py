import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Deteksi Perangkat (GPU NVIDIA via CUDA atau CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

if device.type == "cuda":
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memori GPU yang tersedia: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

# 2. Transformasi & Memuat Dataset MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# 3. Arsitektur Model CNN Sederhana
class SimpleCNN(nn.Module):

  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv_layer = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.fc_layer = nn.Sequential(
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
    )

  def forward(self, x):
    x = self.conv_layer(x)
    x = x.view(x.size(0), -1)  # Flatten
    x = self.fc_layer(x)
    return x


# Inisialisasi model dan pindahkan ke GPU NVIDIA
model = SimpleCNN().to(device)

# 4. Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Proses Pelatihan (Training)
epochs = 3
print("\nMemulai pelatihan model...")

model.train()
for epoch in range(epochs):
  running_loss = 0.0
  for i, (images, labels) in enumerate(train_loader):
    # Pindahkan data ke GPU
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass dan optimisasi
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(
      f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}"
  )

# 6. Evaluasi Model (Testing)
model.eval()
correct = 0
total = 0

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"\nAkurasi model pada data uji: {100 * correct / total:.2f}%")
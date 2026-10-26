import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Definisi Arsitektur Model (Mini-DLSS / Super-Resolution CNN)
class MiniDLSS(nn.Module):
    def __init__(self):
        super(MiniDLSS, self).__init__()
        # Lapisan konvolusi untuk mengekstrak fitur dari gambar resolusi rendah
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        # Lapisan akhir untuk merekonstruksi kembali ke gambar RGB resolusi tinggi
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        # Alur data (Forward Pass) mirip cara kerja AI upscaler
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 2. Inisialisasi Perangkat (GPU NVIDIA CUDA atau CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

model = MiniDLSS().to(device)

# 3. Membuat Dummy Data (Simulasi Gambar Game)
# Target (Ground Truth): Gambar resolusi tinggi ukuran (Batch=1, Channel=3, Tinggi=64, Lebar=64)
high_res_target = torch.randn(1, 3, 64, 64).to(device)

# Input: Gambar resolusi rendah (di-resize/simulasi blur ukuran 32x32 lalu di-upscale kasar ke 64x64)
low_res_input = torch.nn.functional.interpolate(high_res_target, size=(32, 32), mode='bicubic', align_corners=False)
low_res_input = torch.nn.functional.interpolate(low_res_input, size=(64, 64), mode='bicubic', align_corners=False)

# 4. Fungsi Loss dan Optimizer
criterion = nn.MSELoss() # Menghitung selisih pixel antara hasil AI dan target asli
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Proses Training (Simulasi Melatih AI dengan Frame Game)
print("Memulai proses training model Mini-DLSS...")
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    
    # AI menebak/merekonstruksi gambar resolusi tinggi
    outputs = model(low_res_input)
    
    # Hitung error (loss)
    loss = criterion(outputs, high_res_target)
    
    # Backpropagation & update bobot
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

print("Training selesai! Model siap digunakan untuk inferensi game.")

# 6. Tahap Inferensi (Pengujian pada Game)
model.eval()
with torch.no_grad():
    reconstructed_image = model(low_res_input)
    print("Ukuran output rekonstruksi DLSS:", reconstructed_image.shape)
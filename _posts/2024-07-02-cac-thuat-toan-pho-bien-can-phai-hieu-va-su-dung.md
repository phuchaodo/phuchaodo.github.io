---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng


Các thuật toán phổ biến của GAN (Generative Adversarial Networks) là các biến thể và cải tiến của mô hình GAN gốc. Dưới đây là một số thuật toán nổi bật và một ví dụ cụ thể về cách triển khai chúng bằng Python và framework PyTorch.

### 1. GAN (Generative Adversarial Networks)
- **Ý tưởng**: Bao gồm hai mạng: một mạng Generative (G) và một mạng Discriminative (D) cạnh tranh với nhau. Mạng G cố gắng tạo ra dữ liệu giả mạo để lừa mạng D, trong khi mạng D cố gắng phân biệt giữa dữ liệu thật và giả.

**Code ví dụ trong PyTorch**:
```python
import torch
import torch.nn as nn

# Generator definition
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Example usage
z_dim = 100  # Dimension of the latent space
data_dim = 784  # Dimension of the generated data (e.g., MNIST images)
G = Generator(z_dim, data_dim)
D = Discriminator(data_dim)
```

### 2. DCGAN (Deep Convolutional GAN)
- **Ý tưởng**: Sử dụng mạng neural network tích chập sâu cho cả Generator và Discriminator để cải thiện chất lượng ảnh được tạo ra.

**Code ví dụ trong PyTorch**:
```python
import torch
import torch.nn as nn

# Generator definition
class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, hidden_dim=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Example usage
z_dim = 100  # Dimension of the latent space
image_channels = 1  # For grayscale images (e.g., MNIST)
G = Generator(z_dim, image_channels)
D = Discriminator(image_channels)
```

### 3. CGAN (Conditional GAN)
- **Ý tưởng**: Mở rộng GAN để cho phép điều khiển dữ liệu được tạo bằng cách cung cấp thông tin điều kiện (ví dụ: nhãn lớp).

**Code ví dụ trong PyTorch**:
```python
import torch
import torch.nn as nn

# Generator definition
class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        return x

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, input_dim, c_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.fc(x)
        return x

# Example usage
z_dim = 100  # Dimension of the latent space
c_dim = 10  # Dimension of the condition vector (e.g., number of classes)
data_dim = 784  # Dimension of the generated data (e.g., MNIST images)
G = Generator(z_dim, c_dim, data_dim)
D = Discriminator(data_dim, c_dim)
```

Các ví dụ trên chỉ là các mô hình cơ bản. Các thuật toán GAN tiến tiến hơn như WGAN, WGAN-GP, CycleGAN, etc., cũng có thể triển khai tương tự nhưng có thêm các điều chỉnh riêng để cải thiện tính ổn định và chất lượng của mô hình.


Dưới đây là một ví dụ cụ thể về cách triển khai mô hình DCGAN (Deep Convolutional GAN) bằng PyTorch. Trong ví dụ này, chúng ta sẽ sử dụng dữ liệu từ bộ dữ liệu MNIST để huấn luyện mô hình.

### DCGAN (Deep Convolutional GAN)

DCGAN là một biến thể của GAN sử dụng mạng tích chập sâu cho cả Generator và Discriminator để cải thiện chất lượng ảnh được tạo ra. Đây là một trong những thuật toán GAN phổ biến và hiệu quả trong thực tế.

#### Cài đặt mô hình trong PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manual_seed = 999
torch.manual_seed(manual_seed)

# Hyperparameters
batch_size = 128
image_size = 28
z_dim = 100
num_epochs = 50
lr = 0.0002
beta1 = 0.5

# Download MNIST dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Generator definition
class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, hidden_dim=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(z_dim, 1).to(device)
D = Discriminator(1).to(device)

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Binary cross entropy loss and noise
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Train Discriminator
        D.zero_grad()
        label_real = torch.full((batch_size, 1), 1., device=device)
        label_fake = torch.full((batch_size, 1), 0., device=device)

        # Real images
        output = D(real_images)
        errD_real = criterion(output, label_real)
        D_x = output.mean().item()

        # Fake images
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake_images = G(noise)
        output = D(fake_images.detach())
        errD_fake = criterion(output, label_fake)
        D_G_z1 = output.mean().item()

        # Total discriminator loss
        errD = errD_real + errD_fake
        errD.backward()
        optimizer_D.step()

        # Train Generator
        G.zero_grad()
        label_real = torch.full((batch_size, 1), 1., device=device)
        output = D(fake_images)
        errG = criterion(output, label_real)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizer_G.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save generated images
    if epoch == 0:
        vutils.save_image(real_images, '%s/real_samples.png' % "./results", normalize=True)
    
    fake = G(fixed_noise)
    vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % ("./results", epoch + 1), normalize=True)

# Save model checkpoints
torch.save(G.state_dict(), './dcgan_generator.pth')
torch.save(D.state_dict(), './dcgan_discriminator.pth')
```

#### Giải thích code:
- **Generator và Discriminator**: Được định nghĩa bằng lớp `Generator` và `Discriminator` tương ứng. Mạng Generator sử dụng các lớp ConvTranspose2d để chuyển đổi vector nhiễu ngẫu nhiên thành ảnh giả, trong khi Discriminator sử dụng các lớp Conv2d để phân biệt ảnh thật và ảnh giả.
  
- **Optimizer**: Sử dụng Adam optimizer để cập nhật các tham số của Generator và Discriminator.

- **Training Loop**: Vòng lặp huấn luyện với hai giai đoạn chính: huấn luyện Discriminator để phân biệt giữa ảnh thật và ảnh giả và huấn luyện Generator để cố gắng lừa Discriminator bằng cách tạo ra các ảnh giả mà Discriminator cho là ảnh thật.

- **Lưu trữ ảnh và mô hình**: Lưu các ảnh được tạo ra và lưu trữ mô hình Generator cuối cùng sau khi huấn luyện.

Với ví dụ trên, bạn có thể huấn luyện mô hình DCGAN để tạo ra các ảnh chất lượng từ bộ dữ liệu MNIST. Để cải thiện chất lượng hoặc áp dụng cho các bộ dữ liệu khác, bạn có thể điều chỉnh kiến trúc mô hình và các tham số huấn luyện như cách thêm lớp tích chập, kích thước ảnh đầu vào, và số lượng epochs.


Dưới đây là một ví dụ khác về cách triển khai mô hình CGAN (Conditional Generative Adversarial Network) bằng PyTorch. Trong ví dụ này, chúng ta sẽ sử dụng bộ dữ liệu FashionMNIST và huấn luyện mô hình để tạo ra các hình ảnh từ các lớp quần áo khác nhau.

### CGAN (Conditional Generative Adversarial Network)

CGAN mở rộng GAN bằng cách thêm thông tin điều kiện (conditional information), ví dụ như nhãn lớp, để điều khiển quá trình sinh dữ liệu.

#### Cài đặt mô hình trong PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manual_seed = 999
torch.manual_seed(manual_seed)

# Hyperparameters
batch_size = 128
image_size = 28
z_dim = 100
num_epochs = 50
lr = 0.0002
beta1 = 0.5
num_classes = 10  # Number of classes (labels)

# Download FashionMNIST dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Generator definition
class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        return x

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, input_dim, c_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.fc(x)
        return x

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(z_dim, num_classes, image_size * image_size).to(device)
D = Discriminator(image_size * image_size, num_classes).to(device)

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Binary cross entropy loss
criterion = nn.BCELoss()

# Fixed noise for visualization
fixed_noise = torch.randn(num_classes, z_dim, device=device)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels_onehot = torch.zeros(batch_size, num_classes).scatter_(1, labels.view(-1, 1), 1).to(device)

        # Train Discriminator
        D.zero_grad()
        label_real = torch.full((batch_size, 1), 1., device=device)
        label_fake = torch.full((batch_size, 1), 0., device=device)

        # Real images
        output = D(real_images.view(batch_size, -1), labels_onehot)
        errD_real = criterion(output, label_real)
        D_x = output.mean().item()

        # Fake images
        noise = torch.randn(batch_size, z_dim, device=device)
        fake_images = G(noise, labels_onehot)
        output = D(fake_images.detach(), labels_onehot)
        errD_fake = criterion(output, label_fake)
        D_G_z1 = output.mean().item()

        # Total discriminator loss
        errD = errD_real + errD_fake
        errD.backward()
        optimizer_D.step()

        # Train Generator
        G.zero_grad()
        label_real = torch.full((batch_size, 1), 1., device=device)
        output = D(fake_images, labels_onehot)
        errG = criterion(output, label_real)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizer_G.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save generated images
    if epoch == 0:
        vutils.save_image(real_images, '%s/real_samples.png' % "./results", normalize=True)
    
    fake = G(fixed_noise, torch.eye(num_classes, device=device))
    vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % ("./results", epoch + 1), normalize=True)

# Save model checkpoints
torch.save(G.state_dict(), './cgan_generator.pth')
torch.save(D.state_dict(), './cgan_discriminator.pth')
```

#### Giải thích code:
- **Generator và Discriminator**: Được định nghĩa bằng lớp `Generator` và `Discriminator` tương ứng. Mạng Generator nhận vector nhiễu và nhãn lớp để sinh ra ảnh giả. Discriminator nhận ảnh và nhãn lớp để phân biệt giữa ảnh thật và ảnh giả.

- **Optimizer**: Sử dụng Adam optimizer để cập nhật các tham số của Generator và Discriminator.

- **Training Loop**: Vòng lặp huấn luyện với hai giai đoạn chính: huấn luyện Discriminator để phân biệt giữa ảnh thật và ảnh giả và huấn luyện Generator để cố gắng lừa Discriminator bằng cách tạo ra các ảnh giả mà Discriminator cho là ảnh thật.

- **Lưu trữ ảnh và mô hình**: Lưu các ảnh được tạo ra và lưu trữ mô hình Generator cuối cùng sau khi huấn luyện.

Với ví dụ trên, bạn có thể huấn luyện mô hình CGAN để tạo ra các hình ảnh từ bộ dữ liệu FashionMNIST dựa trên nhãn lớp. Để cải thiện chất lượng hoặc áp dụng cho các bộ dữ liệu khác, bạn có thể điều chỉnh kiến trúc mô hình, số lượng epochs, và các tham số huấn luyện khác như cách thay đổi số chiều của vector nhiễu hay số lớp đầu ra.




Hết.

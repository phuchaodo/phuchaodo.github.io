---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng

Trong deep learning, VAE (Variational Autoencoder) là một mô hình generative model phổ biến được sử dụng để học và sinh ra dữ liệu. Dưới đây là một số bước thuật toán chính của VAE:

1. **Encoder (Bộ mã hóa):**
   - Nhận đầu vào là một điểm dữ liệu \( x \) và biến đổi nó thành một phân phối xác suất trên không gian latent \( z \).
   - Dùng mạng nơ-ron để biến đổi \( x \) thành hai vector: mean (\( \mu \)) và log-variance (\( \log\sigma^2 \)) của phân phối Gaussian đa chiều trong không gian latent.

2. **Sampling từ không gian latent:**
   - Sử dụng \( \mu \) và \( \log\sigma^2 \) để lấy mẫu từ phân phối Gaussian \( q(z|x) \) để có được một vector latent \( z \).

3. **Decoder (Bộ giải mã):**
   - Nhận vector latent \( z \) và chuyển đổi nó thành một phân phối xác suất trên không gian dữ liệu \( x \).
   - Dùng mạng nơ-ron để học biến đổi ngược từ \( z \) thành \( x \).

4. **Reconstruction loss (Hàm mất mát tái tạo):**
   - Đo lường sự khác biệt giữa dữ liệu ban đầu \( x \) và dữ liệu tái tạo \( x' \) bằng cách sử dụng hàm loss, thường là hàm lỗi bình phương Euclide.

5. **KL divergence (Kullback-Leibler divergence):**
   - Đo lường sự khác biệt giữa phân phối tiên nghiệm (trước khi xem dữ liệu) \( p(z) \) và phân phối hậu nghiệm (dựa trên dữ liệu) \( q(z|x) \).
   - Đây là thành phần giúp đảm bảo rằng không gian latent được học được phân phối tương tự với phân phối Gaussian chuẩn.

6. **Objective function (Hàm mục tiêu):**
   - Kết hợp hàm mất mát tái tạo và KL divergence để tối ưu hóa mô hình VAE.

Thuật toán VAE tổng quát như vậy và có thể được điều chỉnh để phù hợp với các bài toán cụ thể trong deep learning, nhưng các bước cơ bản như trên là không thay đổi.


Để triển khai mô hình Variational Autoencoder (VAE) bằng PyTorch, chúng ta cần định nghĩa một mạng neural network cho phần encoder và phần decoder của VAE, sau đó xây dựng hàm mất mát và quá trình huấn luyện. Dưới đây là một ví dụ cụ thể về cách triển khai VAE bằng PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 400
epochs = 20
learning_rate = 1e-3

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate VAE model
model = VAE().to(device)

# Loss function: BCE + KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'vae_samples.png')
```

Trong ví dụ trên:

- **Encoder** của VAE là một mạng neural network với hai lớp fully connected (hoàn toàn kết nối) để tính toán mean và log-variance của phân phối Gaussian trong không gian latent.
- **Decoder** của VAE cũng là một mạng neural network với hai lớp fully connected để giải mã từ không gian latent trở lại không gian dữ liệu.
- **Hàm mất mát** bao gồm sự kết hợp của hàm lỗi tái tạo và KL divergence, được sử dụng để đo lường sự khác biệt giữa dữ liệu gốc và dữ liệu tái tạo, đồng thời đảm bảo rằng không gian latent được học một cách hiệu quả.
- **Huấn luyện** được thực hiện qua một vòng lặp đơn giản, trong đó mỗi lượt huấn luyện cập nhật các tham số của mô hình VAE để giảm thiểu hàm mất mát.
- Cuối cùng, một số hình ảnh tái tạo được lưu lại sau khi huấn luyện để kiểm tra chất lượng của mô hình VAE.

Đây là một triển khai cơ bản của VAE bằng PyTorch trên tập dữ liệu MNIST. Bạn có thể thay đổi kiến trúc mạng neural, hàm mất mát, hay các siêu tham số khác để phù hợp với các bài toán và tập dữ liệu khác nhau.


Dưới đây là một ví dụ khác về triển khai mô hình Variational Autoencoder (VAE) bằng PyTorch trên tập dữ liệu FashionMNIST. Ví dụ này sẽ bao gồm một số điều chỉnh nhỏ so với ví dụ trước để phù hợp với đặc điểm của FashionMNIST và cấu trúc mô hình một chút khác:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 256
epochs = 20
learning_rate = 1e-3

# FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate VAE model
model = VAE().to(device)

# Loss function: BCE + KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'vae_fashion_samples.png')
```

Trong ví dụ này:

- Chúng ta sử dụng tập dữ liệu FashionMNIST thay vì MNIST, với các điều chỉnh tương ứng về số lượng lớp ẩn và kích thước các lớp.
- Mô hình VAE vẫn có cấu trúc tương tự như trước, với encoder và decoder được định nghĩa thông qua các lớp fully connected.
- Hàm mất mát vẫn bao gồm sự kết hợp của hàm lỗi tái tạo và KL divergence để đảm bảo mô hình học được không gian latent hiệu quả.
- Quá trình huấn luyện cũng tương tự như trước, với việc lặp lại các batch và cập nhật các tham số của mô hình VAE để giảm thiểu hàm mất mát.

Với ví dụ này, bạn có thể áp dụng mô hình VAE trên tập dữ liệu FashionMNIST và điều chỉnh các siêu tham số để thích nghi với các bài toán khác nhau trong deep learning.


Biến thể của thuật toán Variational Autoencoder (VAE) thường xoay quanh các điều chỉnh và cải tiến để cải thiện hiệu suất và tính ứng dụng của mô hình. Dưới đây là một số biến thể phổ biến của VAE:

1. **Conditional Variational Autoencoder (CVAE)**:
   - Mở rộng VAE để học biểu diễn biểu đồ của dữ liệu có điều kiện trên một nhóm các biến.
   - Cho phép mô hình sinh dữ liệu có điều kiện trên các yếu tố như lớp, nhãn, hoặc các biến môi trường khác.

2. **β-VAE**:
   - Điều chỉnh hàm mất mát của VAE bằng tham số β để cân bằng sự đa dạng và chất lượng của các biểu diễn học được.
   - Giúp điều khiển mức độ sự hiểu biết về không gian biểu diễn so với độ phức tạp của dữ liệu.

3. **Adversarial Variational Bayes (AVB)**:
   - Kết hợp VAE với mạng đối địch (adversarial network) để cải thiện việc ước lượng hàm mật độ xác suất hậu nghiệm của mô hình.
   - Được xây dựng trên lý thuyết Generative Adversarial Networks (GANs), cải thiện chất lượng của dữ liệu tái tạo.

4. **Semi-supervised Variational Autoencoder**:
   - Kết hợp VAE với bài toán học có giám sát để cải thiện khả năng phân loại và biểu diễn của mô hình.
   - Sử dụng nhãn có sẵn trong dữ liệu để học các biểu diễn chất lượng cao hơn và cải thiện khả năng tổng quát hóa của mô hình.

5. **Denoising Variational Autoencoder (DVAE)**:
   - Đối mặt với vấn đề của dữ liệu nhiễu bằng cách sử dụng VAE để học các biểu diễn có khả năng chống nhiễu.
   - Giúp cải thiện khả năng tái tạo của mô hình khi dữ liệu bị nhiễu.

6. **Disentangled Variational Autoencoder**:
   - Mục tiêu là học các biểu diễn không gian latent sao cho các chiều trong không gian latent tương ứng với các thuộc tính riêng biệt của dữ liệu.
   - Hữu ích trong việc phân tích và hiểu các yếu tố quan trọng của dữ liệu, như phân tách biểu tượng và nền tảng trong hình ảnh.

Mỗi biến thể có những điểm mạnh riêng biệt và có thể được áp dụng trong các bối cảnh và bài toán khác nhau trong deep learning và machine learning. Việc lựa chọn biến thể phù hợp phụ thuộc vào mục tiêu cụ thể của bài toán và tính chất của dữ liệu.


Dưới đây là trình bày chi tiết về mỗi biến thể của Variational Autoencoder (VAE) và cách triển khai chúng bằng PyTorch.

### 1. Conditional Variational Autoencoder (CVAE)

Conditional VAE mở rộng VAE bằng cách thêm thông tin điều kiện vào quá trình huấn luyện và sinh dữ liệu. Thông tin điều kiện có thể là nhãn lớp, đặc điểm của dữ liệu, hoặc các biến môi trường khác.

#### Chi tiết thuật toán:

- **Encoder (Bộ mã hóa)**: Nhận vào cả dữ liệu \( x \) và thông tin điều kiện \( y \), biến đổi chúng thành một phân phối xác suất trên không gian latent \( z \).

- **Sampling từ không gian latent**: Sử dụng phân phối xác suất thu được từ bộ mã hóa để lấy mẫu vector latent \( z \).

- **Decoder (Bộ giải mã)**: Nhận vector latent \( z \) và thông tin điều kiện \( y \), chuyển đổi chúng thành một phân phối xác suất trên không gian dữ liệu \( x \).

- **Objective function (Hàm mục tiêu)**: Kết hợp hàm mất mát tái tạo và KL divergence, với sự điều kiện hóa trên thông tin \( y \).

#### Code PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 400
epochs = 20
learning_rate = 1e-3
num_classes = 10  # số lượng lớp trong MNIST

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define Conditional VAE model
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(794, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size + num_classes, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)

    def encode(self, x, y):
        x = x.view(-1, 784)
        y = F.one_hot(y, num_classes=num_classes).float()  # chuyển đổi nhãn lớp thành one-hot encoding
        input_concat = torch.cat((x, y), dim=1)
        h1 = F.relu(self.fc1(input_concat))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        input_concat = torch.cat((z, y), dim=1)
        h3 = F.relu(self.fc3(input_concat))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# Instantiate CVAE model
model = CVAE().to(device)

# Loss function: BCE + KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample_labels = torch.randint(0, num_classes, (64,)).to(device)
    sample = model.decode(sample, sample_labels).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'cvae_samples.png')
```

### 2. β-VAE

β-VAE là một biến thể của VAE được điều chỉnh bởi tham số β, để cân bằng giữa sự đa dạng và chất lượng của biểu diễn học được trong không gian latent.

#### Chi tiết thuật toán:

- **Loss function (Hàm mất mát)**: Điều chỉnh hàm mất mát của VAE bằng cách nhân thêm một hệ số β vào phần KL divergence.
  
  Hàm mất mát tổng quát cho β-VAE:
  \[
  \mathcal{L} = \text{BCE} + \beta \cdot \text{KLD}
  \]
  Trong đó:
  - BCE là binary cross entropy giữa dữ liệu gốc và dữ liệu tái tạo.
  - KLD là KL divergence giữa phân phối tiên nghiệm và phân phối hậu nghiệm trong không gian latent.
  - β là tham số điều chỉnh, quyết định mức độ đa dạng của biểu diễn học được.

#### Code PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 400
epochs = 20
learning_rate = 1e-3
beta = 4  # giá trị tham số β

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define β-VAE model
class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))


        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate β-VAE model
model = BetaVAE().to(device)

# Loss function: BCE + β * KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'beta_vae_samples.png')
```

### 3. Adversarial Variational Bayes (AVB)

AVB kết hợp VAE với mạng đối địch (adversarial network), giúp cải thiện độ phức tạp và chất lượng của biểu diễn học được trong không gian latent.

#### Chi tiết thuật toán:

- **Objective function (Hàm mục tiêu)**: Mục tiêu là cải thiện độ phức tạp của phân phối hậu nghiệm bằng cách sử dụng mạng đối địch để ước lượng một phân phối xấp xỉ tốt hơn cho không gian latent.

- **Mạng đối địch (Adversarial network)**: Được sử dụng để phân biệt giữa các mẫu từ phân phối tiên nghiệm và phân phối hậu nghiệm ước lượng.

#### Code PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 400
epochs = 20
learning_rate = 1e-3

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define VAE model with Adversarial Variational Bayes (AVB)
class AVB_VAE(nn.Module):
    def __init__(self):
        super(AVB_VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)
        
        # Discriminator layers
        self.fc5 = nn.Linear(latent_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def discriminate(self, z):
        h5 = F.relu(self.fc5(z))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.discriminate(z)

# Instantiate AVB_VAE model
model = AVB_VAE().to(device)

# Loss function: BCE for reconstruction, adversarial loss, and KL divergence
def loss_function(recon_x, x, mu, logvar, D_z):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    adversarial_loss = -torch.log(D_z).sum()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + adversarial_loss + KLD

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, D_z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, D_z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'avb_vae_samples.png')
```

### 4. Semi-supervised Variational Autoencoder

Semi-supervised VAE kết hợp VAE với bài toán học có giám sát để cải thiện khả năng phân loại và biểu diễn của mô hình.

#### Chi tiết thuật toán:

- **Objective function (Hàm mục tiêu)**: Kết hợp giữa hàm mất mát của VAE (tái tạo và KL divergence) và hàm mất mát của bài toán phân loại.

- **Training process (Quá trình huấn luyện)**: Bổ sung bộ phân loại vào mô hình VAE, cùng với các chiến lược như label smoothing (làm mượt nhãn) để cải thiện sự tổng quát hóa và khả năng phân loại.

#### Code PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 400
epochs = 20
learning_rate = 1e-3
num_classes = 10  # số lượng lớp trong MNIST

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch

_size, shuffle=True)

# Define Semi-supervised VAE model
class SSVAE(nn.Module):
    def __init__(self):
        super(SSVAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Classifier layer
        
        # Decoder layers
        self.fc4 = nn.Linear(latent_size + num_classes, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1), self.fc3(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        input_concat = torch.cat((z, y), dim=1)
        h4 = F.relu(self.fc4(input_concat))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar, y = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar, F.log_softmax(y, dim=1)

# Instantiate Semi-supervised VAE model
model = SSVAE().to(device)

# Loss function: BCE + KL divergence + CrossEntropy for classification
def loss_function(recon_x, x, mu, logvar, y_pred, y_true):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    CE = F.nll_loss(y_pred, y_true)
    return BCE + KLD + CE

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, y_pred = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, y_pred, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample_labels = torch.randint(0, num_classes, (64,)).to(device)
    sample = model.decode(sample, sample_labels).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'ssvae_samples.png')
```

### 5. Denoising Variational Autoencoder (DVAE)

Denoising VAE giải quyết vấn đề của dữ liệu nhiễu bằng cách học các biểu diễn có khả năng chống nhiễu.

#### Chi tiết thuật toán:

- **Objective function (Hàm mục tiêu)**: Mở rộng hàm mất mát của VAE để bao gồm việc so sánh giữa dữ liệu gốc và dữ liệu đã được làm sạch.

- **Denoising process (Quá trình làm sạch)**: Thêm một quá trình giảm nhiễu vào quá trình huấn luyện, hoặc áp dụng các phương pháp xử lý nhiễu dữ liệu trước khi đưa vào mô hình.

#### Code PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 20
hidden_size = 400
epochs = 20
learning_rate = 1e-3

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define Denoising VAE model
class DVAE(nn.Module):
    def __init__(self):
        super(DVAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate Denoising VAE model
model = DVAE().to(device)

# Loss function: BCE + KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data + torch.randn_like(data) * 0.3)  # Add Gaussian noise to input
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'dvae_samples.png')
```

### 6. Disentangled Variational Autoencoder

Disentangled VAE học các biểu diễn trong không gian latent sao cho mỗi chiều tương ứng với một thuộc tính đặc trưng của dữ liệu.

#### Chi tiết thuật toán:

- **Objective function (Hàm mục tiêu)**: Bổ sung thêm các điều kiện ràng buộc vào hàm mục tiêu của VAE để đảm bảo các chiều trong không gian latent là độc lập và biểu diễn rõ ràng các thuộc tính của dữ liệu.

- **Structural constraints (Ràng buộc cấu trúc)**: Áp dụng các ràng buộc đối với biể

u diễn trong không gian latent (ví dụ: ràng buộc độ lệch chuẩn gần với 1).

#### Code PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_size = 10
hidden_size = 400
epochs = 20
learning_rate = 1e-3

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define Disentangled VAE model
class DsVAE(nn.Module):
    def __init__(self):
        super(DsVAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate Disentangled VAE model
model = DsVAE().to(device)

# Loss function: BCE + KL divergence + structural constraints
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    struct_constr = torch.mean((mu**2 + torch.exp(logvar) - logvar - 1))
    return BCE + KLD + struct_constr

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Main training loop
for epoch in range(1, epochs + 1):
    train(epoch)

# Save some reconstructed images
with torch.no_grad():
    sample = torch.randn(64, latent_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'dsvae_samples.png')
```

Mỗi loại VAE đã thúc đẩy nghiên cứu phát triển mô hình học sâu và nhận diện chuyển động, ánh xạ hình ảnh.



Hết.

---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng

Cả hai thuật toán MobileNet và EfficientNet đều là các mô hình deep learning được phát triển để giảm kích thước và tính toán của mạng neural network, đặc biệt là trong bối cảnh các ứng dụng nhúng trên thiết bị có tài nguyên hạn chế như điện thoại thông minh.

### 1. MobileNet

**Giải thích:** MobileNet là một mạng neural network được thiết kế để có thể hoạt động hiệu quả trên các thiết bị di động và các thiết bị nhúng khác. Nó sử dụng một kỹ thuật gọi là Depthwise Separable Convolution để giảm số lượng tham số và tính toán so với các mạng truyền thống.

**Code PyTorch:**
```python
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['MobileNetV1', 'mobilenet_v1']

model_urls = {
    'mobilenet_v1': 'https://download.pytorch.org/models/mobilenet_v1.pth.tar',
}

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        # Define layers according to MobileNetV1 architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenet_v1(pretrained=False, progress=True, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

**Ví dụ sử dụng:**
```python
# Load pretrained MobileNetV1 model
model = mobilenet_v1(pretrained=True)

# Set model to evaluation mode
model.eval()

# Example input tensor (batch_size, channels, height, width)
example_input = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(example_input)
print(output.shape)  # Shape of the output tensor
```

### 2. EfficientNet

**Giải thích:** EfficientNet là một mạng neural network được tối ưu hóa để đạt được một hiệu suất tốt hơn với số lượng tham số và tính toán thấp hơn so với các mạng truyền thống. Nó sử dụng một kỹ thuật gọi là Compound Scaling để tự động cân bằng kích thước mạng theo chiều rộng, chiều sâu và độ phân giải của hình ảnh đầu vào.

**Code PyTorch:**
```python
from efficientnet_pytorch import EfficientNet

# Define EfficientNet model (e.g., EfficientNet-B0)
model = EfficientNet.from_name('efficientnet-b0')

# Alternatively, load pretrained weights
# model = EfficientNet.from_pretrained('efficientnet-b0')

# Set model to evaluation mode
model.eval()

# Example input tensor (batch_size, channels, height, width)
example_input = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(example_input)
print(output.shape)  # Shape of the output tensor
```

**Ví dụ sử dụng:**
```python
# Load pretrained EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Set model to evaluation mode
model.eval()

# Example input tensor (batch_size, channels, height, width)
example_input = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(example_input)
print(output.shape)  # Shape of the output tensor
```

Đoạn code cho cả MobileNet và EfficientNet mô tả cách để định nghĩa và sử dụng các mô hình này trong PyTorch. Bạn có thể thay đổi kích thước của input và số lượng lớp đầu ra tuỳ thuộc vào nhu cầu của bạn.



Hết.

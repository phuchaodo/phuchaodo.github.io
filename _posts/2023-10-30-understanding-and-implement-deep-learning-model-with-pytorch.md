---
title: '[Note] Understanding and Implementing Deep Learning Models with PyTorch'
date: 2023-10-30
permalink: /posts/2023/10/30/understanding-and-implement-deep-learning-model-with-pytorch/
tags:
  - research
  - deep learning
  - convolutional
  - neural networks
---

Hiểu và cài đặt mô hình deep learning với pytorch

Understanding and Implementing Deep Learning Models with PyTorch
======
 
## Cơ chế hoạt động

Để hiểu cách hoạt động của mỗi block trong một mô hình deep learning, bạn cần nắm vững các khái niệm về các lớp (layers) và các phép toán (operations) được sử dụng trong mỗi lớp đó. Đồng thời, bạn cũng cần hiểu cách các lớp này tương tác với nhau để tạo thành một mô hình hoàn chỉnh.

Đối với việc thực hiện các phép toán trong mỗi block, đây là một số bước bạn có thể thử:

1. **Đọc tài liệu và mã nguồn mẫu**: Đầu tiên, nên đọc tài liệu về mô hình và các lớp được sử dụng. Nếu có sẵn mã nguồn mẫu (ví dụ: trên GitHub hoặc trong tài liệu hướng dẫn), hãy đọc mã để hiểu cách mà phép toán được thực hiện.

2. **Tìm hiểu về các thư viện và framework**: Học cách sử dụng các thư viện phổ biến như PyTorch hoặc TensorFlow. Đây là các công cụ mạnh mẽ giúp bạn xây dựng và huấn luyện mô hình deep learning.

3. **Thực hành với ví dụ đơn giản**: Bắt đầu từ các ví dụ đơn giản để thực hiện các phép toán cơ bản. Ví dụ: học cách tạo một lớp fully connected, một lớp convolutional, hoặc một lớp recurrent.

4. **Tham gia vào các khóa học hoặc khoá học trực tuyến**: Nếu bạn đang bắt đầu, việc tham gia vào các khóa học hoặc khoá học trực tuyến về deep learning sẽ giúp bạn hiểu rõ hơn về các khái niệm cơ bản và cách thực hiện chúng.

5. **Đọc mã nguồn của các mô hình sẵn có**: Nếu có thể, hãy tìm và đọc mã nguồn của các mô hình deep learning đã được triển khai bởi cộng đồng. Điều này sẽ giúp bạn hiểu cách các block được kết hợp lại với nhau.

6. **Thực hành và tùy chỉnh mã nguồn**: Hãy thử tạo một mô hình đơn giản và thực hiện các phép toán trong đó. Tiến dần từ việc xây dựng các lớp cơ bản đến việc kết hợp chúng để tạo ra một mô hình hoàn chỉnh.

Đối với việc sử dụng PyTorch, đây là một số bước cơ bản:

1. **Cài đặt PyTorch**: Đầu tiên, bạn cần cài đặt PyTorch trên máy tính của mình. Hướng dẫn cài đặt có thể tìm thấy trên trang web chính thức của PyTorch.

2. **Import thư viện**: Sau khi cài đặt, bạn cần import PyTorch vào mã của mình bằng câu lệnh `import torch`.

3. **Tạo các lớp (layers)**: Sử dụng PyTorch để tạo các lớp cần thiết cho mô hình của bạn. Ví dụ: `nn.Linear` cho lớp fully connected, `nn.Conv2d` cho lớp convolutional, và như vậy.

4. **Kết hợp các lớp**: Sử dụng các lớp đã tạo để xây dựng mô hình của bạn. Kết hợp các lớp này bằng cách sử dụng các phép toán toán học hoặc hàm kích hoạt.

5. **Huấn luyện mô hình**: Sử dụng dữ liệu huấn luyện và thuật toán tối ưu hóa để huấn luyện mô hình của bạn.

Dưới đây là một ví dụ về cách tạo một mô hình đơn giản sử dụng PyTorch:

```python
import torch
import torch.nn as nn

# Định nghĩa một mô hình đơn giản với 1 lớp fully connected
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # Một lớp fully connected với input size là 10 và output size là 5

    def forward(self, x):
        x = self.fc(x)
        return x

# Tạo một instance của mô hình
model = SimpleModel()

# Tạo input và thực hiện forward pass
input_data = torch.randn(3, 10)  # Tạo tensor 3x10 ngẫu nhiên
output = model(input_data)

print(output)
```


## Kiến trúc CNN (Convolutional Neural Network) 

Nó là một loại mô hình deep learning đặc biệt thiết kế cho việc xử lý dữ liệu lưới như hình ảnh. CNN sử dụng các lớp convolutional để trích xuất đặc trưng và giúp mô hình học được các mẫu cục bộ trong dữ liệu.

Dưới đây, tôi sẽ giải thích chi tiết về kiến trúc CNN từng block và các phép toán được thực hiện trong mỗi block.

### Block 1: Convolutional Layer

#### Convolutional Operation:
- **Input**: Một ảnh hoặc một feature map từ layer trước.
- **Filter (Kernel)**: Một ma trận nhỏ được sử dụng để quét (convolve) qua ảnh input.
- **Stride**: Số bước mà filter di chuyển qua ảnh.
- **Padding**: Thêm các hàng và cột 0 xung quanh ảnh input để duy trì kích thước.

#### Cách code trong PyTorch:

```python
import torch.nn as nn

# Định nghĩa một convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

### Block 2: Activation Function (ReLU)

#### Activation Function:
- **Input**: Đầu ra từ convolutional layer.
- **Công dụng**: Đưa vào một phép toán phi tuyến tính để mô hình có thể học các mối quan hệ phức tạp hơn giữa các đặc trưng.

#### Cách code trong PyTorch:

```python
import torch.nn as nn

# Sử dụng ReLU activation
activation = nn.ReLU()
```

### Block 3: Pooling Layer

#### Pooling Operation:
- **Input**: Đầu ra từ activation layer.
- **Công dụng**: Giảm kích thước không gian của đầu ra, giúp giảm thiểu số lượng tham số và tính toán.

#### Cách code trong PyTorch:

```python
import torch.nn as nn

# Định nghĩa một max pooling layer
pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
```

### Block 4: Fully Connected Layer (FC Layer)

#### Fully Connected Operation:
- **Input**: Vector hoặc tensor đã được làm phẳng từ layer trước.
- **Công dụng**: Kết nối toàn bộ các đặc trưng được trích xuất từ các convolutional layers để đưa ra dự đoán cuối cùng.

#### Cách code trong PyTorch:

```python
import torch.nn as nn

# Định nghĩa một fully connected layer
fc_layer = nn.Linear(in_features=256, out_features=10)  # Ví dụ: có 256 đầu vào và 10 đầu ra (số lớp đầu ra)
```

### Cách kết hợp các block:

```python
import torch.nn as nn

# Định nghĩa một mô hình CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=16 * 16 * 16, out_features=10)  # Ví dụ: 16x16 là kích thước sau pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)  # Làm phẳng tensor
        x = self.fc(x)
        return x

# Tạo một instance của mô hình
model = SimpleCNN()
```

## Thực hiện train, validate, testing

Để thực hiện việc train, validate và testing trên một tập dataset, chúng ta cần làm các bước sau:

1. **Load Dataset**: Chọn một tập dataset (ví dụ: CIFAR-10) và tải nó vào trong môi trường làm việc.

2. **Chuẩn bị Dữ liệu**: Chuẩn bị dữ liệu train, validation và test set, và tiến hành tiền xử lý (nếu cần).

3. **Xây dựng Mô hình**: Định nghĩa một mô hình CNN (hoặc sử dụng mô hình đã có).

4. **Chọn Loss Function và Optimizer**: Chọn hàm mất mát và thuật toán tối ưu hóa.

5. **Huấn luyện Mô hình**: Sử dụng dữ liệu huấn luyện để huấn luyện mô hình.

6. **Kiểm tra trên Tập Validation**: Đánh giá mô hình trên tập validation để theo dõi hiệu suất.

7. **Kiểm tra trên Tập Test**: Đánh giá mô hình trên tập test để đánh giá hiệu suất cuối cùng.

8. **Hiển thị Kết Quả**: Hiển thị thông tin như confusion matrix, accuracy và loss qua các epoch.

Dưới đây là một ví dụ cụ thể sử dụng tập dữ liệu CIFAR-10 và PyTorch:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Step 1: Load Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Step 3: Build Model
class SimpleCNN(nn.Module):
    # ...

# Step 4: Choose Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Step 5: Train the Model
# (Loop through epochs, batches, and perform forward, backward, and update steps)

# Step 6: Validate the Model
# (Evaluate on validation set)

# Step 7: Test the Model
# (Evaluate on test set)

# Step 8: Display Results
# (Confusion matrix, accuracy, loss values)
```

### Step 5: Train the Model

```python
# Assume model, criterion, optimizer đã được định nghĩa và trainloader đã được chuẩn bị.

num_epochs = 5  # Số lượng epochs (lặp lại qua toàn bộ tập dữ liệu)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader)}')
```

### Step 6: Validate the Model

```python
# Assume model và testloader đã được chuẩn bị.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy}%')
```

### Step 7: Test the Model

```python
# Assume model và testloader đã được chuẩn bị.

confusion_matrix = torch.zeros(10, 10)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print('Confusion Matrix:')
print(confusion_matrix)
```

### Step 8: Display Results

Để hiển thị kết quả trong PyTorch, bạn có thể sử dụng các thư viện như `matplotlib` để vẽ biểu đồ loss và accuracy qua từng epoch.

```python
import matplotlib.pyplot as plt

# Định nghĩa list để lưu lại thông tin
losses = []
accuracies = []

# ... (trong vòng lặp huấn luyện, sau khi tính loss và update mô hình)
losses.append(running_loss / len(trainloader))

# ... (sau khi kiểm tra trên tập test)
accuracies.append(accuracy)

# Vẽ biểu đồ
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')

plt.subplot(1,2,2)
plt.plot(accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')

plt.show()
```

## Chi tiết cài đặt một mô hình CNN đơn giản

### Block 1: Convolutional Layer

#### Mô tả:
- **Công dụng**: Convolutional layer giúp trích xuất đặc trưng cục bộ từ ảnh hoặc feature map trước đó bằng cách sử dụng các filters (kernels).
- **Biểu diễn Toán**:
   - **Input**: Feature map hoặc ảnh từ layer trước đó (kích thước: C x H x W).
   - **Filter (Kernel)**: Một ma trận nhỏ được sử dụng để quét (convolve) qua ảnh input (kích thước: C_in x K x K).
   - **Stride**: Số bước mà filter di chuyển qua ảnh.
   - **Padding**: Thêm các hàng và cột 0 xung quanh ảnh input để duy trì kích thước.

#### Biểu diễn Toán (Convolution):
   - Đầu ra: Feature map mới (kích thước: C_out x H_out x W_out).
   - H_out, W_out: Chiều cao và chiều rộng của feature map sau convolution.

### Block 2: Activation Function (ReLU)

#### Mô tả:
- **Công dụng**: Activation function (ReLU) được áp dụng sau convolutional layer để đưa vào một phép toán phi tuyến tính và kích hoạt các đặc trưng.

#### Biểu diễn Toán (ReLU):
   - Đầu ra: Feature map đã được kích hoạt (cùng kích thước với đầu vào).

### Block 3: Pooling Layer

#### Mô tả:
- **Công dụng**: Pooling layer giúp giảm kích thước không gian của đầu ra, giúp giảm thiểu số lượng tham số và tính toán.

#### Biểu diễn Toán (Max Pooling):
   - **Input**: Feature map sau activation (kích thước: C x H x W).
   - **Kernel Size**: Kích thước của cửa sổ pooling.
   - **Stride**: Số bước mà pooling window di chuyển qua feature map.

   - **Output**: Feature map sau pooling (kích thước: C x H_out x W_out).

### Block 4: Fully Connected Layer (FC Layer)

#### Mô tả:
- **Công dụng**: Fully connected layer kết nối toàn bộ các đặc trưng đã được trích xuất từ các convolutional layers để đưa ra dự đoán cuối cùng.

#### Biểu diễn Toán:
   - **Input**: Vector hoặc tensor đã được làm phẳng từ layer trước.
   - **Weights (W)**: Ma trận trọng số kết nối các đầu vào đến các đầu ra (kích thước: num_inputs x num_outputs).
   - **Bias (b)**: Vector bias được cộng thêm vào kết quả của phép toán tuyến tính.

   - **Output**: Vector dự đoán (kích thước: num_outputs).
   


Hết.

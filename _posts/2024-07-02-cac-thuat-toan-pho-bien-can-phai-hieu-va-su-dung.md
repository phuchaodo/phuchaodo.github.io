---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng

Mô hình lan truyền (diffusion model) là một lớp mô hình thống kê được sử dụng để mô tả sự lan truyền của các đối tượng, thông tin, hay sự kiện trong một mạng xã hội, môi trường hay bất kỳ hệ thống nào khác có tính chất lan truyền. Các biến thể của mô hình này có thể bao gồm:

1. **Mô hình SIR**: Phân biệt giữa các cá nhân là S (susceptible - dễ bị lây nhiễm), I (infected - đã bị lây nhiễm), và R (recovered - đã hồi phục hoặc chết). Mô hình này giúp mô tả sự lây lan của các bệnh truyền nhiễm.

2. **Mô hình SI**: Đơn giản hơn mô hình SIR, chỉ phân biệt giữa các cá nhân là S và I, mô tả sự lây lan trong các bệnh có tính chất lây lan nhanh.

3. **Mô hình tuyến tính lan truyền (Linear Threshold Model)**: Mô hình mô tả sự lan truyền thông tin trong mạng xã hội, trong đó mỗi cá nhân có ngưỡng chấp nhận thông tin, khi mà số lượng người đồng ý vượt quá ngưỡng này thì cá nhân sẽ chấp nhận thông tin đó.

4. **Mô hình ngưỡng Schelling (Schelling Threshold Model)**: Mô hình mô tả sự phân tán của các cá nhân trong một không gian mà mỗi cá nhân có một ngưỡng phù hợp mô tả sự chịu đựng của họ với các cá nhân khác.

Để cài đặt một mô hình lan truyền bằng PyTorch, bạn cần xây dựng một mạng neural network hoặc mô hình tính toán trên một đồ thị hoặc một ma trận. Dưới đây là một ví dụ đơn giản về mô hình SIR sử dụng PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SIRModel(nn.Module):
    def __init__(self, num_nodes, beta, gamma):
        super(SIRModel, self).__init__()
        self.num_nodes = num_nodes
        self.beta = beta  # Tỷ lệ lây nhiễm
        self.gamma = gamma  # Tỷ lệ hồi phục
        
        # Trạng thái ban đầu (mỗi node S, I hoặc R)
        self.states = nn.Parameter(torch.randint(0, 3, size=(num_nodes,)))

    def forward(self):
        # Tính toán sự thay đổi của mỗi node theo mô hình SIR
        infected_nodes = torch.nonzero(self.states == 1).squeeze()
        susceptible_nodes = torch.nonzero(self.states == 0).squeeze()
        
        # Tính toán lây nhiễm
        for i in infected_nodes:
            prob_infection = torch.rand(self.num_nodes)
            infected_neighbors = torch.nonzero(prob_infection < self.beta).squeeze()
            self.states[infected_neighbors] = 1
        
        # Tính toán hồi phục
        for i in infected_nodes:
            prob_recovery = torch.rand(self.num_nodes)
            recovered_nodes = torch.nonzero(prob_recovery < self.gamma).squeeze()
            self.states[recovered_nodes] = 2

# Sử dụng mô hình
num_nodes = 100
beta = 0.3
gamma = 0.1
model = SIRModel(num_nodes, beta, gamma)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Huấn luyện mô hình
for epoch in range(100):
    optimizer.zero_grad()
    model.forward()
    loss = ...  # Định nghĩa hàm mất mát (nếu cần)
    loss.backward()
    optimizer.step()

# Trích xuất kết quả cuối cùng
final_states = model.states
```

Trong ví dụ trên, mô hình SIR được triển khai như một lớp PyTorch `SIRModel`, trong đó `states` là trạng thái của mỗi nút (0=S, 1=I, 2=R). Trong `forward()`, chúng ta tính toán sự thay đổi của các trạng thái dựa trên các tham số `beta` và `gamma`, và sau đó sử dụng optimizer để cập nhật trọng số của mô hình.


Dưới đây là thêm một số ví dụ về các biến thể của mô hình lan truyền và cách cài đặt chúng bằng PyTorch:

### 1. Mô hình SI (Susceptible-Infected)

Mô hình này mô tả sự lây lan của một thông tin, ý tưởng hoặc bệnh trong cộng đồng mà không có khả năng hồi phục.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SIModel(nn.Module):
    def __init__(self, num_nodes, beta):
        super(SIModel, self).__init__()
        self.num_nodes = num_nodes
        self.beta = beta  # Tỷ lệ lây nhiễm
        
        # Trạng thái ban đầu (mỗi node S hoặc I)
        self.states = nn.Parameter(torch.randint(0, 2, size=(num_nodes,)))

    def forward(self):
        # Tính toán sự thay đổi của mỗi node theo mô hình SI
        infected_nodes = torch.nonzero(self.states == 1).squeeze()
        susceptible_nodes = torch.nonzero(self.states == 0).squeeze()
        
        # Tính toán lây nhiễm
        for i in infected_nodes:
            prob_infection = torch.rand(self.num_nodes)
            infected_neighbors = torch.nonzero(prob_infection < self.beta).squeeze()
            self.states[infected_neighbors] = 1

# Sử dụng mô hình
num_nodes = 100
beta = 0.2
model_si = SIModel(num_nodes, beta)
optimizer = optim.Adam(model_si.parameters(), lr=0.01)

# Huấn luyện mô hình
for epoch in range(100):
    optimizer.zero_grad()
    model_si.forward()
    # Có thể định nghĩa hàm mất mát nếu cần
    # loss = ...
    # loss.backward()
    optimizer.step()

# Trích xuất kết quả cuối cùng
final_states_si = model_si.states
```

### 2. Mô hình Linear Threshold Model

Mô hình này mô tả sự lan truyền thông tin trong mạng xã hội, trong đó mỗi cá nhân có một ngưỡng chấp nhận thông tin.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LinearThresholdModel(nn.Module):
    def __init__(self, adjacency_matrix, thresholds):
        super(LinearThresholdModel, self).__init__()
        self.adjacency_matrix = adjacency_matrix  # Ma trận kề của đồ thị
        self.thresholds = thresholds  # Ngưỡng của mỗi cá nhân
        
        # Trạng thái ban đầu (mỗi node chưa chấp nhận thông tin)
        self.states = nn.Parameter(torch.zeros(len(thresholds), dtype=torch.float))

    def forward(self):
        # Tính toán sự thay đổi của mỗi node theo mô hình Linear Threshold
        new_states = torch.zeros_like(self.states)
        for i in range(len(self.states)):
            if self.states[i] == 0:
                neighbors = torch.nonzero(self.adjacency_matrix[i]).squeeze()
                influence = torch.sum(self.states[neighbors] / len(neighbors))
                if influence >= self.thresholds[i]:
                    new_states[i] = 1
                else:
                    new_states[i] = 0
            else:
                new_states[i] = self.states[i]
        self.states.data = new_states

# Sử dụng mô hình
adjacency_matrix = torch.tensor([[0, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 0, 0],
                                 [0, 1, 0, 0]])
thresholds = torch.tensor([0.5, 0.6, 0.4, 0.7])
model_ltm = LinearThresholdModel(adjacency_matrix, thresholds)
optimizer = optim.Adam(model_ltm.parameters(), lr=0.01)

# Huấn luyện mô hình
for epoch in range(100):
    optimizer.zero_grad()
    model_ltm.forward()
    # Có thể định nghĩa hàm mất mát nếu cần
    # loss = ...
    # loss.backward()
    optimizer.step()

# Trích xuất kết quả cuối cùng
final_states_ltm = model_ltm.states
```

### 3. Mô hình Schelling Threshold Model

Mô hình này mô tả sự phân tán của các cá nhân trong một không gian dựa trên mức độ chấp nhận của họ đối với các cá nhân khác.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SchellingModel(nn.Module):
    def __init__(self, grid, thresholds):
        super(SchellingModel, self).__init__()
        self.grid = nn.Parameter(grid)  # Lưới các cá nhân
        self.thresholds = thresholds  # Ngưỡng của mỗi cá nhân
        
    def forward(self):
        # Tính toán sự thay đổi của mỗi cá nhân theo mô hình Schelling Threshold
        new_grid = self.grid.clone()
        for i in range(self.grid.size(0)):
            for j in range(self.grid.size(1)):
                if self.grid[i, j] != 0:
                    neighbors = self.get_neighbors(i, j)
                    similar_neighbors = torch.sum(self.grid[neighbors] == self.grid[i, j])
                    if similar_neighbors < self.thresholds[self.grid[i, j]]:
                        # Tìm vị trí ngẫu nhiên để di chuyển
                        empty_positions = torch.nonzero(self.grid == 0)
                        if empty_positions.size(0) > 0:
                            random_empty = empty_positions[torch.randint(0, empty_positions.size(0), (1,))].squeeze()
                            new_grid[random_empty[0], random_empty[1]] = self.grid[i, j]
                            new_grid[i, j] = 0
        self.grid.data = new_grid

    def get_neighbors(self, i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid.size(0) and 0 <= nj < self.grid.size(1):
                    neighbors.append((ni, nj))
        return neighbors

# Sử dụng mô hình
grid = torch.tensor([[1, 2, 0],
                     [0, 1, 2],
                     [2, 0, 1]])
thresholds = torch.tensor([2, 2, 2])  # Mỗi loại cá nhân có ngưỡng riêng
model_schelling = SchellingModel(grid, thresholds)
optimizer = optim.Adam(model_schelling.parameters(), lr=0.01)

# Huấn luyện mô hình
for epoch in range(100):
    optimizer.zero_grad()
    model_schelling.forward()
    # Có thể định nghĩa hàm mất mát nếu cần
    # loss = ...
    # loss.backward()
    optimizer.step()

# Trích xuất kết quả cuối cùng
final_grid_schelling = model_schelling.grid
```

Các ví dụ trên giúp bạn hiểu rõ hơn về cách cài đặt các mô hình lan truyền khác nhau bằng PyTorch, từ mô hình SIR cho đến các mô hình lan truyền thông tin và mô hình Schelling cho sự phân tán không gian của các cá nhân. Điều này có thể giúp bạn áp dụng chúng vào các bài toán thực tế khác nhau như nghiên cứu mạng xã hội, dịch bệnh hay phân tán xã hội.




Hết.

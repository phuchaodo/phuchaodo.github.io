---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng

Các thuật toán phổ biến của mạng nơ-ron đồ thị (Graph Neural Networks - GNNs) bao gồm nhiều phương pháp khác nhau, nhưng hai phương pháp cơ bản nhất là Graph Convolutional Network (GCN) và Graph Attention Network (GAT). Dưới đây là một trình bày chi tiết về cách thực hiện và mã nguồn Python sử dụng thư viện PyTorch cho hai thuật toán này.

### 1. Graph Convolutional Network (GCN)

GCN là một trong những mô hình đơn giản nhất và phổ biến nhất trong lớp mạng nơ-ron đồ thị. Nó sử dụng phép tích chập trên đồ thị để cập nhật các đặc trưng của đỉnh.

**Mã nguồn Python sử dụng PyTorch cho GCN:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

**Giải thích:**
- `GCNConv`: Lớp convolutional trên đồ thị từ thư viện `torch_geometric.nn`.
- `GCN` là một `nn.Module` trong PyTorch, bao gồm hai lớp convolution (`conv1` và `conv2`) và các lớp activation và dropout giữa chúng.
- `forward` function thực hiện lan truyền thuận của mô hình, sử dụng hàm kích hoạt ReLU và hàm dropout giữa các lớp.

### 2. Graph Attention Network (GAT)

GAT là một biến thể nâng cao của GCN, tập trung vào sự chú ý của các đỉnh trong đồ thị để cải thiện hiệu suất.

**Mã nguồn Python sử dụng PyTorch cho GAT:**

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, num_heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_size, heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, num_classes, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU activation
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

**Giải thích:**
- `GATConv`: Lớp attention trên đồ thị từ `torch_geometric.nn`.
- `GAT` là một `nn.Module` trong PyTorch, bao gồm hai lớp attention (`conv1` và `conv2`) với số lượng head (num_heads).
- `forward` function thực hiện lan truyền thuận của mô hình, sử dụng hàm kích hoạt ELU và hàm dropout giữa các lớp.

### Lưu ý:
- Cả hai mô hình đều sử dụng `torch_geometric`, một thư viện hỗ trợ các phép toán trên đồ thị trong PyTorch.
- Mã nguồn trên chỉ là ví dụ cơ bản để giới thiệu cách triển khai GCN và GAT. Các siêu tham số như số lượng lớp, kích thước ẩn, số lượng head, hàm kích hoạt, và xử lý dữ liệu có thể được điều chỉnh để phù hợp với bài toán cụ thể của bạn.


Ngoài Graph Convolutional Network (GCN) và Graph Attention Network (GAT), còn có nhiều thuật toán khác trong lớp mạng nơ-ron đồ thị (Graph Neural Networks - GNNs), mỗi thuật toán có những đặc điểm riêng biệt và cách hoạt động khác nhau. Dưới đây là một số thuật toán phổ biến khác và mã nguồn Python sử dụng PyTorch cho mỗi thuật toán đó.

### 1. GraphSAGE (Graph Sample and Aggregate)

GraphSAGE là một phương pháp mở rộng cho các mô hình GNN bằng cách lấy mẫu và tổng hợp thông tin từ các hàng xóm của mỗi đỉnh.

**Mã nguồn Python sử dụng PyTorch cho GraphSAGE:**

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.conv2 = SAGEConv(hidden_size, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)
```

### 2. Gated Graph Neural Networks (GGNN)

GGNN sử dụng cơ chế cổng để điều chỉnh cách thông tin được truyền tải trong mạng nơ-ron đồ thị, phù hợp cho các tác vụ yêu cầu xử lý chuỗi dữ liệu trên đồ thị.

**Mã nguồn Python sử dụng PyTorch cho GGNN:**

```python
from torch_geometric.nn import GatedGraphConv

class GGNN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GGNN, self).__init__()
        self.conv1 = GatedGraphConv(num_features, hidden_size)
        self.conv2 = GatedGraphConv(hidden_size, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)
```

### 3. Graph Isomorphism Network (GIN)

GIN tập trung vào tính toán độ tương tự giữa các đồ thị để học các biểu diễn đồ thị không phụ thuộc vào cấu trúc.

**Mã nguồn Python sử dụng PyTorch cho GIN:**

```python
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        ))
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)
```

### Lưu ý:

- Các mô hình trên sử dụng các lớp convolutional trên đồ thị từ `torch_geometric.nn` và có cấu trúc tương tự như GCN và GAT.
- Mỗi thuật toán có những điểm mạnh và yếu khác nhau, phù hợp với các bài toán và dữ liệu khác nhau.
- Để triển khai các thuật toán này, bạn cần cài đặt thư viện `torch_geometric` và xử lý dữ liệu đồ thị phù hợp trước khi đưa vào mô hình.


Để minh họa cách thực hiện các thuật toán mạng nơ-ron đồ thị (GNNs) cho các bài toán cụ thể như phân loại đỉnh (node classification) và phân loại đồ thị (graph classification), chúng ta sẽ sử dụng các ví dụ cụ thể và mã nguồn Python sử dụng thư viện PyTorch và Torch Geometric.

### 1. Node Classification

#### Bài toán:
Phân loại các đỉnh trong đồ thị thành các lớp nhất định dựa trên các đặc trưng của đỉnh và cấu trúc của đồ thị.

#### Mã nguồn Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

# Load dataset (cora dataset as an example)
dataset = Planetoid(root='data/', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GCN(num_features=dataset.num_features, hidden_size=16, num_classes=dataset.num_classes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training
def train_model(model, data, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

train_model(model, data, optimizer, criterion)

# Evaluation
def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        print(f'Test Accuracy: {acc:.4f}')

evaluate_model(model, data)
```

#### Giải thích:
- Đoạn mã trên sử dụng tập dữ liệu Cora từ `torch_geometric.datasets.Planetoid` làm ví dụ. Tập dữ liệu Cora bao gồm các đặc trưng đỉnh, cạnh và nhãn lớp cho mỗi đỉnh.
- Mô hình `GCN` được triển khai bằng cách sử dụng lớp `GCNConv` từ `torch_geometric.nn`, bao gồm hai lớp convolutional trên đồ thị và hàm kích hoạt ReLU.
- Quá trình huấn luyện và đánh giá mô hình được thực hiện thông qua hàm `train_model` và `evaluate_model`.

### 2. Graph Classification

#### Bài toán:
Phân loại toàn bộ đồ thị thành các lớp nhất định dựa trên các đặc trưng toàn cục của đồ thị.

#### Mã nguồn Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

# Load dataset (MUTAG dataset as an example)
dataset = TUDataset(root='data/', name='MUTAG', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = scatter_mean(x, batch, dim=0)  # Global pooling over the nodes in each graph
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GCN(num_features=dataset.num_node_features, hidden_size=32, num_classes=dataset.num_classes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training
def train_model(model, loader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(loader.dataset)}')

train_model(model, loader, optimizer, criterion)

# Evaluation
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    print(f'Test Accuracy: {correct / total:.4f}')

evaluate_model(model, loader)
```

#### Giải thích:
- Đoạn mã trên sử dụng tập dữ liệu MUTAG từ `torch_geometric.datasets.TUDataset` làm ví dụ. Tập dữ liệu này chứa các đồ thị nhỏ với đặc trưng nút và nhãn lớp đồ thị.
- Mô hình `GCN` được triển khai với lớp convolutional trên đồ thị và pooling toàn cục (`scatter_mean`) để tính toán đặc trưng toàn cục của đồ thị.
- Quá trình huấn luyện và đánh giá mô hình được thực hiện thông qua hàm `train_model` và `evaluate_model`, với việc sử dụng `DataLoader` để tải và chia dữ liệu thành các batch.

### Lưu ý:
- Để chạy các ví dụ trên, bạn cần cài đặt thư viện `torch`, `torch_geometric`, và `torch_scatter`.
- Các mô hình và cài đặt có thể được điều chỉnh tùy thuộc vào yêu cầu cụ thể của bài toán và tập dữ liệu.
- Các ví dụ trên chỉ là một phần nhỏ trong một loạt các ứng dụng của mạng nơ-ron đồ thị, và có thể được mở rộng và tinh chỉnh để phù hợp với các tác vụ khác nhau.


Dưới đây là thêm một số ví dụ khác về cách thực hiện mạng nơ-ron đồ thị (GNNs) cho các bài toán cụ thể khác nhau như dự đoán cạnh (link prediction) và phân cụm đồ thị (graph clustering), sử dụng thư viện PyTorch và Torch Geometric.

### 3. Link Prediction

#### Bài toán:
Dự đoán sự tồn tại hoặc không tồn tại của các cạnh giữa các cặp đỉnh trong đồ thị.

#### Mã nguồn Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

# Load dataset (Karate club dataset as an example)
dataset = KarateClub()
data = dataset[0]

class LinkPredictionGCN(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(LinkPredictionGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize model
model = LinkPredictionGCN(num_features=dataset.num_features, hidden_size=16)

# Edge indices for positive and negative samples
pos_edge_index = data.edge_index[:, data.train_pos_edge_index]
neg_edge_index = data.edge_index[:, data.train_neg_edge_index]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
def train_link_prediction(model, pos_edge_index, neg_edge_index, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_output = model(data.x, pos_edge_index)
        neg_output = model(data.x, neg_edge_index)
        pos_scores = (pos_output[0, :] * pos_output[1, :]).sum(dim=-1)
        neg_scores = (neg_output[0, :] * neg_output[1, :]).sum(dim=-1)
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        targets = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).to(scores.device)
        loss = F.binary_cross_entropy_with_logits(scores, targets)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

train_link_prediction(model, pos_edge_index, neg_edge_index, optimizer)

# Evaluation (using ROC AUC score)
def evaluate_link_prediction(model, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        pos_output = model(data.x, pos_edge_index)
        neg_output = model(data.x, neg_edge_index)
        pos_scores = (pos_output[0, :] * pos_output[1, :]).sum(dim=-1).sigmoid().cpu().numpy()
        neg_scores = (neg_output[0, :] * neg_output[1, :]).sum(dim=-1).sigmoid().cpu().numpy()
        y_true = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]).numpy()
        y_pred = np.concatenate([pos_scores, neg_scores])
        roc_auc = roc_auc_score(y_true, y_pred)
        print(f'ROC AUC Score: {roc_auc:.4f}')

evaluate_link_prediction(model, data.test_pos_edge_index, data.test_neg_edge_index)
```

#### Giải thích:
- Đoạn mã trên sử dụng tập dữ liệu Karate Club từ `torch_geometric.datasets.KarateClub` làm ví dụ. Đây là một tập dữ liệu nhỏ với đồ thị của một câu lạc bộ karate, mục đích là dự đoán cạnh giữa các cặp đỉnh.
- Mô hình `LinkPredictionGCN` được triển khai bằng cách sử dụng hai lớp `GCNConv` từ `torch_geometric.nn`.
- Quá trình huấn luyện và đánh giá được thực hiện thông qua hàm `train_link_prediction` và `evaluate_link_prediction`, sử dụng hàm mất mát `binary_cross_entropy_with_logits` và đánh giá bằng ROC AUC score.

### 4. Graph Clustering

#### Bài toán:
Phân cụm các đỉnh trong đồ thị thành các nhóm dựa trên cấu trúc của đồ thị.

#### Mã nguồn Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset (MUTAG dataset as an example)
dataset = TUDataset(root='data/', name='MUTAG', use_node_attr=True)
data = dataset[0]

class GraphClusteringGCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_clusters):
        super(GraphClusteringGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_clusters)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GraphClusteringGCN(num_features=dataset.num_node_features, hidden_size=32, num_clusters=2)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training
def train_graph_clustering(model, data, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

train_graph_clustering(model, data, optimizer, criterion)

# Visualization of graph clustering (using NetworkX)
def visualize_graph_clustering(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)
        node_color = pred.cpu().numpy()
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, node_color=node_color, cmap=plt.cm.tab10, with_labels=True, node_size=300, font_size=10)
        plt.title('Graph Clustering')
        plt.show()

visualize_graph_clustering(model, data)
```

#### Giải thích:
- Đoạn mã trên sử dụng tập dữ liệu MUTAG từ `torch_geometric.datasets.TUDataset` làm ví dụ. Tập dữ liệu này chứa các đồ thị nhỏ với đặc trưng nút và nhãn lớp đồ thị.
- Mô hình `GraphClusteringGCN` được triển khai với lớp convolutional trên đồ thị và một lớp fully connected để phân cụm đồ thị thành hai nhóm.
- Quá trình huấn luyện và đánh giá được thực hiện thông qua hàm `train_graph_clustering` và `visualize_graph_clustering`, sử dụng NetworkX để trực quan hóa kết quả phân cụm đồ thị.

### Lưu ý:
- Các ví dụ trên giới thiệu cách thực hiện các thuật toán GNNs cho các bài toán cụ thể như dự đoán cạnh và phân cụm đồ thị.
- Mỗi ví dụ có thể được điều chỉnh để phù hợp với yêu cầu cụ thể của bài toán và tập dữ liệu.
- Để chạy các ví dụ này, cần cài đặt các thư viện `torch`, `torch_geometric`, `networkx` và `matplotlib`.




Dưới đây là thêm một ví dụ khác về cách thực hiện mạng nơ-ron đồ thị (GNN) cho bài toán phát hiện cộng đồng trong đồ thị, sử dụng thư viện PyTorch và Torch Geometric.

### 5. Community Detection (Phát hiện cộng đồng)

#### Bài toán:
Phân tách các đỉnh trong đồ thị thành các cộng đồng hoặc nhóm dựa trên cấu trúc của đồ thị.

#### Mã nguồn Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset (Karate club dataset as an example)
dataset = KarateClub()
data = dataset[0]

class CommunityDetectionGCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_communities):
        super(CommunityDetectionGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_communities)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
model = CommunityDetectionGCN(num_features=dataset.num_features, hidden_size=16, num_communities=2)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training
def train_community_detection(model, data, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

train_community_detection(model, data, optimizer, criterion)

# Visualization of community detection (using NetworkX)
def visualize_community_detection(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)
        node_color = pred.cpu().numpy()
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, node_color=node_color, cmap=plt.cm.tab10, with_labels=True, node_size=300, font_size=10)
        plt.title('Community Detection')
        plt.show()

visualize_community_detection(model, data)
```

#### Giải thích:
- Đoạn mã trên sử dụng tập dữ liệu Karate Club từ `torch_geometric.datasets.KarateClub` làm ví dụ. Tập dữ liệu này bao gồm một đồ thị đơn giản của một câu lạc bộ karate với các đỉnh và cạnh.
- Mô hình `CommunityDetectionGCN` được triển khai với lớp convolutional trên đồ thị (`GCNConv`) để phân tách các đỉnh thành hai cộng đồng.
- Quá trình huấn luyện và đánh giá được thực hiện thông qua hàm `train_community_detection` và `visualize_community_detection`, sử dụng NetworkX để trực quan hóa kết quả phân tách cộng đồng của đồ thị.

### Lưu ý:
- Các ví dụ trên giới thiệu cách thực hiện mạng nơ-ron đồ thị (GNN) cho các bài toán khác nhau như phát hiện cộng đồng trong đồ thị.
- Mỗi ví dụ có thể được điều chỉnh để phù hợp với yêu cầu cụ thể của bài toán và tập dữ liệu.
- Để chạy các ví dụ này, cần cài đặt các thư viện `torch`, `torch_geometric`, `networkx` và `matplotlib`.




Hết.

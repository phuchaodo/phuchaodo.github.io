---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng


Được, dưới đây là một số loại biểu đồ phổ biến được tạo bằng thư viện `matplotlib` trong Python, kết hợp với `torch` (một thư viện phổ biến trong deep learning).

Trước khi bắt đầu, bạn cần cài đặt các thư viện cần thiết nếu chưa có:

```bash
pip install matplotlib torch
```

### 1. Biểu đồ đường (Line Plot)

Biểu đồ đường thường được sử dụng để biểu diễn sự thay đổi của một biến theo thời gian hoặc không gian.

```python
import matplotlib.pyplot as plt
import torch

# Tạo dữ liệu giả định
x = torch.arange(0, 10, 0.1)
y = torch.sin(x)

# Vẽ biểu đồ đường
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), y.numpy(), label='sin(x)')
plt.title('Biểu đồ hàm sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Biểu đồ điểm (Scatter Plot)

Biểu đồ điểm thường được sử dụng để thể hiện mối quan hệ giữa hai biến.

```python
# Tạo dữ liệu giả định
x = torch.randn(100)
y = 2*x + torch.randn(100)

# Vẽ biểu đồ điểm
plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), color='blue', label='Dữ liệu')
plt.title('Biểu đồ điểm')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Biểu đồ cột (Bar Chart)

Biểu đồ cột thường được sử dụng để so sánh giữa các nhóm.

```python
# Dữ liệu giả định
categories = ['A', 'B', 'C', 'D', 'E']
values = torch.tensor([23, 45, 56, 32, 78])

# Vẽ biểu đồ cột
plt.figure(figsize=(8, 6))
plt.bar(categories, values.numpy(), color='green')
plt.title('Biểu đồ cột')
plt.xlabel('Nhóm')
plt.ylabel('Giá trị')
plt.grid(True)
plt.show()
```

### 4. Biểu đồ hình tròn (Pie Chart)

Biểu đồ hình tròn thường được sử dụng để hiển thị tỷ lệ phần trăm của các thành phần.

```python
# Dữ liệu giả định
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']

# Vẽ biểu đồ hình tròn
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Biểu đồ hình tròn')
plt.show()
```

### 5. Biểu đồ boxplot

Biểu đồ boxplot thường được sử dụng để mô tả phân phối của dữ liệu và phát hiện giá trị ngoại lệ.

```python
# Dữ liệu giả định
data = torch.randn(100)

# Vẽ biểu đồ boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(data.numpy())
plt.title('Biểu đồ boxplot')
plt.ylabel('Giá trị')
plt.grid(True)
plt.show()
```

Đây là một số ví dụ cơ bản về cách sử dụng `matplotlib` để vẽ các loại biểu đồ phổ biến, kết hợp với `torch` để tạo dữ liệu. Bạn có thể thay đổi dữ liệu và tùy chỉnh biểu đồ theo nhu cầu của mình.


Dưới đây là thêm một vài loại biểu đồ khác và các ví dụ cụ thể, sử dụng `matplotlib` và `torch` trong Python.

### 6. Biểu đồ violin plot

Biểu đồ violin plot thường được sử dụng để hiển thị phân phối của dữ liệu và so sánh giữa các nhóm.

```python
# Dữ liệu giả định
data1 = torch.randn(100)
data2 = torch.randn(100) + 2  # Shift mean

# Vẽ biểu đồ violin plot
plt.figure(figsize=(8, 6))
plt.violinplot([data1.numpy(), data2.numpy()], showmeans=False, showmedians=True)
plt.title('Biểu đồ violin plot')
plt.xlabel('Dữ liệu')
plt.ylabel('Giá trị')
plt.xticks([1, 2], ['Dữ liệu 1', 'Dữ liệu 2'])
plt.grid(True)
plt.show()
```

### 7. Biểu đồ đồ thị (Graph Plot)

Biểu đồ đồ thị thường được sử dụng để biểu diễn mối quan hệ giữa các đỉnh (nodes) và các cạnh (edges).

```python
import networkx as nx

# Tạo đồ thị giả định
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])

# Vẽ biểu đồ đồ thị
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_color='black')
plt.title('Biểu đồ đồ thị')
plt.show()
```

### 8. Biểu đồ 2D Density Plot

Biểu đồ 2D density plot thường được sử dụng để hiển thị mật độ phân bố của dữ liệu trong không gian hai chiều.

```python
# Dữ liệu giả định
x = torch.randn(1000)
y = 2*x + torch.randn(1000)

# Vẽ biểu đồ 2D density plot
plt.figure(figsize=(8, 6))
plt.hist2d(x.numpy(), y.numpy(), bins=30, cmap='Blues')
plt.colorbar(label='Mật độ')
plt.title('Biểu đồ 2D density')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

### 9. Biểu đồ heatmap

Biểu đồ heatmap thường được sử dụng để hiển thị dữ liệu dưới dạng màu sắc trên một lưới.

```python
# Dữ liệu giả định
data = torch.randn(10, 10)

# Vẽ biểu đồ heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data.numpy(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Biểu đồ heatmap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

### 10. Biểu đồ đường dạng nối (Connected Line Plot)

Biểu đồ đường dạng nối thường được sử dụng để biểu diễn dữ liệu chuỗi liên tục.

```python
# Dữ liệu giả định
x = torch.arange(0, 10, 0.1)
y1 = torch.sin(x)
y2 = torch.cos(x)

# Vẽ biểu đồ đường dạng nối
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), y1.numpy(), marker='o', linestyle='-', color='b', label='sin(x)')
plt.plot(x.numpy(), y2.numpy(), marker='s', linestyle='--', color='r', label='cos(x)')
plt.title('Biểu đồ đường dạng nối')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 11. Biểu đồ radar chart

Biểu đồ radar chart thường được sử dụng để hiển thị dữ liệu đa chiều dưới dạng một đa giác.

```python
# Dữ liệu giả định
categories = ['A', 'B', 'C', 'D', 'E']
values1 = torch.tensor([4, 3, 2, 5, 4])
values2 = torch.tensor([5, 1, 3, 4, 2])

# Vẽ biểu đồ radar chart
plt.figure(figsize=(8, 6))

angles = torch.linspace(0, 2 * torch.pi, len(categories), endpoint=False).numpy()
values1 = torch.cat((values1, values1[0].unsqueeze(0)))
values2 = torch.cat((values2, values2[0].unsqueeze(0)))

ax = plt.subplot(111, polar=True)
ax.plot(angles, values1.numpy(), marker='o', linestyle='-', label='Series 1')
ax.fill(angles, values1.numpy(), alpha=0.25)

ax.plot(angles, values2.numpy(), marker='s', linestyle='-', label='Series 2')
ax.fill(angles, values2.numpy(), alpha=0.25)

ax.set_theta_offset(torch.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)

plt.title('Biểu đồ radar chart')
plt.legend()
plt.show()
```

Đây là một số ví dụ thú vị và phổ biến về cách sử dụng `matplotlib` để vẽ các loại biểu đồ khác nhau, kết hợp với `torch` để tạo và xử lý dữ liệu. Bạn có thể thử nghiệm và tùy chỉnh các ví dụ này để phù hợp với nhu cầu và dữ liệu của mình.


Dưới đây là thêm một số ví dụ khác về cách sử dụng `matplotlib` và `torch` để vẽ các loại biểu đồ khác nhau trong Python.

### 12. Biểu đồ hexbin

Biểu đồ hexbin thường được sử dụng để hiển thị mật độ của dữ liệu hai chiều.

```python
# Dữ liệu giả định
x = torch.randn(1000)
y = 2*x + torch.randn(1000)

# Vẽ biểu đồ hexbin
plt.figure(figsize=(8, 6))
plt.hexbin(x.numpy(), y.numpy(), gridsize=30, cmap='Blues')
plt.colorbar(label='Mật độ')
plt.title('Biểu đồ hexbin')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

### 13. Biểu đồ đường 3D

Biểu đồ đường 3D thường được sử dụng để biểu diễn dữ liệu trong ba chiều.

```python
from mpl_toolkits.mplot3d import Axes3D

# Dữ liệu giả định
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y)
Z = torch.sin(torch.sqrt(X**2 + Y**2))

# Vẽ biểu đồ đường 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis')
ax.set_title('Biểu đồ đường 3D')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
```

### 14. Biểu đồ bar 3D

Biểu đồ bar 3D thường được sử dụng để biểu diễn dữ liệu dưới dạng cột trong ba chiều.

```python
# Dữ liệu giả định
x = ['A', 'B', 'C', 'D', 'E']
y = torch.tensor([23, 45, 56, 32, 78])

# Vẽ biểu đồ bar 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.bar(x, y.numpy(), color='skyblue')
ax.set_title('Biểu đồ bar 3D')
ax.set_xlabel('Nhóm')
ax.set_ylabel('Giá trị')
ax.set_zlabel('Độ cao')
plt.show()
```

### 15. Biểu đồ stacked bar

Biểu đồ stacked bar thường được sử dụng để biểu diễn các thành phần phân tầng của dữ liệu.

```python
# Dữ liệu giả định
categories = ['A', 'B', 'C', 'D']
values1 = torch.tensor([25, 30, 35, 20])
values2 = torch.tensor([15, 25, 20, 10])

# Vẽ biểu đồ stacked bar
plt.figure(figsize=(8, 6))
plt.bar(categories, values1.numpy(), label='Nhóm 1', color='blue')
plt.bar(categories, values2.numpy(), bottom=values1.numpy(), label='Nhóm 2', color='orange')
plt.title('Biểu đồ stacked bar')
plt.xlabel('Nhóm')
plt.ylabel('Giá trị')
plt.legend()
plt.grid(True)
plt.show()
```

### 16. Biểu đồ đa giao nhau (Venn diagram)

Biểu đồ đa giao nhau thường được sử dụng để biểu diễn mối quan hệ giữa các tập hợp.

```python
from matplotlib_venn import venn2

# Dữ liệu giả định
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Vẽ biểu đồ Venn
plt.figure(figsize=(8, 6))
venn2([set1, set2], ('Set 1', 'Set 2'))
plt.title('Biểu đồ Venn')
plt.show()
```

### 17. Biểu đồ quạt (Fan chart)

Biểu đồ quạt thường được sử dụng để biểu diễn phạm vi của dự đoán hoặc ước tính.

```python
# Dữ liệu giả định
x = torch.arange(1, 11)
y = torch.randn(10)
error = torch.randn(10) * 2

# Vẽ biểu đồ quạt
plt.figure(figsize=(8, 6))
plt.errorbar(x.numpy(), y.numpy(), yerr=error.numpy(), fmt='o', color='green', ecolor='red', elinewidth=2, capsize=4)
plt.title('Biểu đồ quạt')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

### 18. Biểu đồ định kỳ (Polar plot)

Biểu đồ định kỳ thường được sử dụng để biểu diễn dữ liệu theo hệ tọa độ polar.

```python
# Dữ liệu giả định
theta = torch.linspace(0, 2 * torch.pi, 100)
r = torch.abs(torch.sin(2 * theta))

# Vẽ biểu đồ định kỳ
plt.figure(figsize=(8, 6))
plt.polar(theta.numpy(), r.numpy())
plt.title('Biểu đồ định kỳ')
plt.show()
```

Đây là một số ví dụ tiếp theo về cách sử dụng `matplotlib` và `torch` để vẽ các loại biểu đồ phổ biến khác nhau. Bạn có thể tham khảo và thử nghiệm để tìm hiểu thêm về các tính năng và tùy chỉnh của từng loại biểu đồ.
Hết.

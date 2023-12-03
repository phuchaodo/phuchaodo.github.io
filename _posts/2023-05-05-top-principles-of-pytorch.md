---
title: '[Note] Top principles of Pytorch'
date: 2023-05-05
permalink: /posts/2023/05/05/top-principles-of-pytorch/
tags:
  - research
  - python
  - pytorch
--- 

Principles of Pytorch
======

Tensors in PyTorch are multi-dimensional arrays. They are similar to NumPy's ndarrays but can run on GPUs.

```python
import torch

# Create a 2x3 tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)
```

PyTorch uses dynamic computation graphs, meaning the graph is built on-the-fly as operations are executed. This provides flexibility for modifying the graph during runtime.

```python
# Define two tensors
a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([3.], requires_grad=True)

# Compute result
c = a * b
c.backward()

# Gradients
print(a.grad)  # Gradient w.r.t a
```

PyTorch allows easy switching between CPU and GPU. Utilize .to(device) for optimal performance.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = tensor.to(device)
```

PyTorch's autograd provides automatic differentiation for all operations on tensors. Set requires_grad=True to track computations.

```python
x = torch.tensor([2.], requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Gradient of y w.r.t x
```

PyTorch provides the nn.Module class to define neural network architectures. Create custom layers by subclassing.

```python
import torch.nn as nn

class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.fc(x)
```

PyTorch offers various predefined layers, loss functions, and optimization algorithms in the nn module.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

For efficient data handling and batching, PyTorch offers the Dataset and DataLoader classes.

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    # ... (methods to define)
    
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Typically, training in PyTorch follows the pattern: forward pass, compute loss, backward pass, and parameter update.

```python
for epoch in range(epochs):
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

Save and load your models using torch.save() and torch.load().

```python
# Save
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model.load_state_dict(torch.load('model_weights.pth'))
```

While PyTorch operates in eager mode by default, it offers Just-In-Time (JIT) compilation for production-ready models.

```python
scripted_model = torch.jit.script(model)
scripted_model.save("model_jit.pt")
```

Link tham khảo
======

[Link 1](https://medium.com/@kasperjuunge/10-principles-of-pytorch-bbe4bf0c42cd)


Hết.

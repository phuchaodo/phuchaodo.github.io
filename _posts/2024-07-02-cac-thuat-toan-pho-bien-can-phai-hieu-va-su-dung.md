---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng

Trong deep learning, Transformer là một mô hình cực kỳ quan trọng được sử dụng rộng rãi cho các bài toán liên quan đến xử lý ngôn ngữ tự nhiên (NLP), như dịch máy và sinh văn bản. Mô hình Transformer giúp giải quyết các vấn đề mà các mô hình trước đây như RNN và CNN gặp phải, như vấn đề về vanishing gradient và khả năng chia sẻ thông tin song song.

### Các thành phần chính của Transformer

1. **Self-Attention Mechanism**: Cơ chế này cho phép mô hình "tập trung" vào các phần quan trọng của đầu vào khi tính toán đầu ra. Đây là thành phần chủ yếu giúp Transformer xử lý các mối quan hệ dài hạn và dài ngắn trong chuỗi đầu vào.

2. **Multi-head Attention**: Để nâng cao khả năng học của mô hình, Transformer sử dụng nhiều đầu attention độc lập song song, sau đó kết hợp kết quả từ các đầu attention này.

3. **Feedforward Neural Networks**: Mỗi lớp trong mô hình Transformer bao gồm một lớp feedforward, giúp cải thiện độ phức tạp của biểu diễn đầu ra.

4. **Normalization Layers**: Để ổn định quá trình huấn luyện, Transformer sử dụng các lớp normalization như Layer Normalization và Batch Normalization.

### Code ví dụ sử dụng PyTorch

Dưới đây là một ví dụ đơn giản về cách triển khai một mô hình Transformer Encoder sử dụng PyTorch. Mô hình sẽ có một Encoder Layer gồm Self-Attention, Feedforward Layer và các lớp normalization.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"
        
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, _ = query.shape
        Q = query
        K = key
        V = value
        
        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])  # (batch_size, num_heads, seq_length, seq_length)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        x = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )
        
        x = self.fc_out(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))
        return x

# Example usage:
embed_size = 512
num_heads = 8
ff_hidden_size = 2048
dropout = 0.1
batch_size = 16
seq_length = 20

# Initialize encoder layer
encoder_layer = EncoderLayer(embed_size, num_heads, ff_hidden_size, dropout)

# Dummy input tensor
input_tensor = torch.randn(batch_size, seq_length, embed_size)

# Dummy mask (you may need to define a proper mask based on your task)
mask = torch.ones((batch_size, 1, seq_length, seq_length))

# Forward pass through encoder layer
output_tensor = encoder_layer(input_tensor, mask)
print(output_tensor.shape)  # Output shape: torch.Size([16, 20, 512])
```

Trong ví dụ trên, chúng ta triển khai một Encoder Layer của Transformer bao gồm một MultiHeadAttention layer và một FeedForward layer. Mỗi lớp đều được bao bọc bởi một Layer Normalization và Dropout để cải thiện khả năng học và ổn định quá trình huấn luyện. Lưu ý rằng mask được sử dụng để ẩn các vị trí không cần thiết trong attention.


Dưới đây là một ví dụ cụ thể hơn về cách triển khai một mô hình Transformer hoàn chỉnh để giải quyết bài toán dịch máy đơn giản bằng PyTorch. Trong ví dụ này, chúng ta sẽ sử dụng một Transformer Encoder và một Transformer Decoder để thực hiện bản dịch từ tiếng Anh sang tiếng Pháp.

### Chuẩn bị dữ liệu

Đầu tiên, chúng ta cần chuẩn bị dữ liệu. Trong ví dụ này, để đơn giản, chúng ta sẽ sử dụng bộ dữ liệu dịch máy nhỏ được tạo từ các cặp câu tiếng Anh - tiếng Pháp.

```python
# Install torchtext if you haven't already
# !pip install torchtext

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

# Load the dataset
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'

# Create Field objects
SRC = Field(tokenize = 'spacy', 
            tokenizer_language='en_core_web_sm', 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TGT = Field(tokenize = 'spacy', 
            tokenizer_language='fr_core_news_sm', 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

# Load data
train_data, valid_data, test_data = Multi30k.splits(exts = ('.en', '.fr'), 
                                                    fields = (SRC, TGT))

# Build vocabulary
SRC.build_vocab(train_data, min_freq = 2)
TGT.build_vocab(train_data, min_freq = 2)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Định nghĩa mô hình Transformer

Tiếp theo, chúng ta sẽ định nghĩa mô hình Transformer sử dụng các lớp Transformer Encoder và Transformer Decoder.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        self.transformer_encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src shape: (src_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        
        for layer in self.transformer_encoder_layers:
            src = layer(src, src_mask)
        
        return src

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        self.transformer_decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg shape: (trg_len, batch_size)
        trg = self.embedding(trg) * math.sqrt(self.embed_dim)
        trg = self.pos_encoder(trg)
        
        for layer in self.transformer_decoder_layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        return output
```

### Huấn luyện mô hình

Cuối cùng, chúng ta sẽ huấn luyện mô hình trên dữ liệu và đánh giá kết quả.

```python
# Define parameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TGT.vocab)
EMB_DIM = 256
N_HEADS = 8
FF_HID_DIM = 512
N_LAYERS = 3
DROPOUT = 0.1

# Define encoder and decoder
encoder = TransformerEncoder(INPUT_DIM, EMB_DIM, N_HEADS, FF_HID_DIM, N_LAYERS, DROPOUT).to(device)
decoder = TransformerDecoder(OUTPUT_DIM, EMB_DIM, N_HEADS, FF_HID_DIM, N_LAYERS, DROPOUT).to(device)

# Initialize weights
def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

encoder.apply(init_weights)
decoder.apply(init_weights)

# Define optimizer and criterion
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index = TGT.vocab.stoi[TGT.pad_token])

# Training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        src_mask = model.make_src_mask(src)
        trg_mask = model.make_trg_mask(trg)
        
        enc_src = model.encoder(src, src_mask)
        output = model.decoder(trg, enc_src, trg_mask, src_mask)
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            src_mask = model.make_src_mask(src)
            trg_mask = model.make_trg_mask(trg)

            enc_src = model.encoder(src, src_mask)
            output = model.decoder(trg, enc_src, trg_mask, src_mask)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Define iterators
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)

# Training
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(encoder, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(encoder, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(encoder.state_dict(), 'encoder.pt')
        torch.save(decoder.state_dict(), 'decoder.pt')
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Testing
encoder.load_state_dict(torch.load('encoder.pt'))
decoder.load_state_dict(torch.load('decoder.pt'))

test_loss = evaluate(encoder, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} |')
```

Trong ví dụ này:

- Chúng ta sử dụng dữ liệu từ bộ dữ liệu Multi30k của TorchText để tạo bộ dữ liệu dịch máy tiếng Anh - Pháp.
- Định nghĩa một mô hình Transformer bao gồm Transformer Encoder và Transformer Decoder.
- Huấn luyện mô hình trên dữ liệu huấn luyện và đánh giá kết quả trên tập validation.
- Lưu lại mô hình có kết quả tốt nhất và đánh giá trên tập test.

Ví dụ này cung cấp một hướ

ng dẫn cơ bản về cách triển khai một mô hình Transformer cho bài toán dịch máy bằng PyTorch. Để tối ưu hơn và phù hợp với bài toán cụ thể, bạn có thể điều chỉnh các siêu tham số, thêm các tính năng như attention visualization, và nâng cao khả năng đối phó với overfitting.


Có, mô hình Transformer không chỉ được sử dụng cho các bài toán xử lý ngôn ngữ tự nhiên như dịch máy hay sinh văn bản mà còn có thể áp dụng cho các bài toán prediction và classification trong các lĩnh vực khác như computer vision và các dạng dữ liệu có cấu trúc khác. Dưới đây là một ví dụ về việc sử dụng mô hình Transformer cho bài toán image classification sử dụng PyTorch và Hugging Face Transformers library.

### Sử dụng mô hình Transformer cho bài toán image classification

Trong ví dụ này, chúng ta sẽ sử dụng một mô hình Transformer pre-trained là `ViT` (Vision Transformer) từ thư viện Hugging Face Transformers để phân loại ảnh trên tập dữ liệu CIFAR-10.

#### Chuẩn bị dữ liệu

```python
import torch
import torchvision
import torchvision.transforms as transforms

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT requires input images to be of size 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

#### Định nghĩa và huấn luyện mô hình Transformer (ViT)

```python
from transformers import ViTModel, ViTForImageClassification, ViTFeatureExtractor
from torch import nn, optim
import torch.nn.functional as F

# Load pre-trained ViT model
model_name = 'google/vit-base-patch16-224-in21k'
vit_model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)  # 10 classes in CIFAR-10

# Freeze all layers except the last classifier layer
for param in vit_model.parameters():
    param.requires_grad = False

# Replace the classifier layer with a new one for CIFAR-10
vit_model.classifier = nn.Linear(vit_model.config.hidden_size, 10)

# Define optimizer and loss function
optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)

# Training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = correct / total
    return epoch_loss / len(train_loader), accuracy

# Evaluation loop
def evaluate(model, test_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return epoch_loss / len(test_loader), accuracy

# Training and evaluation
num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train(vit_model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(vit_model, test_loader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
```

Trong ví dụ này:

- Chúng ta sử dụng mô hình `ViTForImageClassification` từ thư viện Transformers để phân loại ảnh từ CIFAR-10 dataset.
- Mô hình được huấn luyện trên tập dữ liệu huấn luyện và đánh giá trên tập dữ liệu kiểm tra.
- Chúng ta sử dụng optimizer Adam và hàm mất mát CrossEntropyLoss cho quá trình huấn luyện.
- Mỗi epoch, chúng ta in ra các chỉ số như loss và accuracy trên cả tập huấn luyện và tập kiểm tra.

Việc sử dụng mô hình Transformer (ở đây là ViT) cho bài toán image classification thể hiện tính linh hoạt và hiệu quả của kiến trúc này, không chỉ giới hạn trong các bài toán xử lý ngôn ngữ tự nhiên mà còn mở rộng ra nhiều lĩnh vực khác.


Dưới đây là một ví dụ cụ thể hơn về cách triển khai một mô hình Transformer cho bài toán classification và prediction sử dụng PyTorch và thư viện Hugging Face Transformers. Trong ví dụ này, chúng ta sẽ sử dụng mô hình Transformer BERT (Bidirectional Encoder Representations from Transformers) để thực hiện phân loại văn bản.

### Chuẩn bị dữ liệu

Trong ví dụ này, chúng ta sẽ sử dụng tập dữ liệu Sentiment Analysis từ IMDb với mục đích phân loại review là tích cực (positive) hoặc tiêu cực (negative).

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Apply tokenizer to dataset
encoded_dataset = dataset.map(tokenize_batch, batched=True)

# Prepare DataLoader
train_loader = DataLoader(encoded_dataset['train'], batch_size=16, shuffle=True)
test_loader = DataLoader(encoded_dataset['test'], batch_size=16, shuffle=False)
```

### Định nghĩa và huấn luyện mô hình Transformer (BERT)

```python
# Define BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
)

# Training
trainer.train()

# Evaluation
trainer.evaluate()
```

Trong ví dụ này:

- Chúng ta sử dụng mô hình BERT pre-trained từ thư viện Transformers để thực hiện phân loại văn bản.
- Dữ liệu được chuẩn bị và đưa vào DataLoader để huấn luyện và đánh giá.
- Mô hình được huấn luyện trong vài epochs trên tập dữ liệu huấn luyện và đánh giá trên tập dữ liệu kiểm tra.
- Sử dụng optimizer AdamW và hàm mất mát cross-entropy loss cho quá trình huấn luyện.

Ví dụ này minh họa cách sử dụng mô hình Transformer (BERT) cho bài toán classification với dữ liệu văn bản, nó có thể áp dụng tương tự cho các bài toán prediction khác, ví dụ như phân loại ảnh, dự đoán chuỗi thời gian, hoặc các bài toán khác có cấu trúc dữ liệu tương tự.


Dưới đây là một ví dụ khác về cách sử dụng mô hình Transformer cho bài toán prediction, cụ thể là dự đoán giá cổ phiếu sử dụng mô hình Transformer Encoder-Decoder. Trong ví dụ này, chúng ta sẽ xây dựng một mô hình Transformer để dự đoán giá cổ phiếu dựa trên dữ liệu lịch sử.

### Chuẩn bị dữ liệu

Trong ví dụ này, chúng ta sẽ sử dụng dữ liệu lịch sử giá cổ phiếu của Apple từ Yahoo Finance.

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('AAPL.csv')

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Select closing price
data = df[['Close']].values.astype(float)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Convert to PyTorch tensors
tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
```

### Định nghĩa mô hình Transformer Encoder-Decoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        self.transformer_encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        for layer in self.transformer_encoder_layers:
            src = layer(src, src_mask)
        
        return src

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(output_dim, embed_dim)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)
        
        self.transformer_decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg)
        trg = self.pos_decoder(trg)
        
        for layer in self.transformer_decoder_layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        return output

# Define model dimensions
input_dim = 1  # Univariate time series (closing price)
output_dim = 1  # Predicting one-step ahead
embed_dim = 64
num_heads = 4
ff_hidden_dim = 128
num_layers = 3
dropout = 0.1

# Initialize models
encoder = TransformerEncoder(input_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout)
decoder = TransformerDecoder(output_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout)

# Define optimizer and loss function
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# Split data into train and test sets
train_size = int(len(tensor_data) * 0.8)
train_data = tensor_data[:train_size]
test_data = tensor_data[train_size:]

# Training loop
def train_model(encoder, decoder, train_data, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        
        for i in range(len(train_data) - 1):
            src = train_data[i:i+1]
            trg = train_data[i+1:i+2]
            
            optimizer.zero_grad()
            
            src_mask = None  # No masking needed for autoregressive prediction
            trg_mask = None
            
            enc_src = encoder(src, src_mask)
            output = decoder(trg, enc_src, trg_mask, src_mask)
            
            loss = criterion(output, trg)
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Training the model
train_model(encoder, decoder, train_data, optimizer, criterion, num_epochs=100)

# Evaluation loop
def evaluate_model(encoder, decoder, test_data):
    encoder.eval()
    decoder.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(test_data) - 1):
            src = test_data[i:i+1]
            trg = test_data[i+1:i+2]
            
            src_mask = None
            trg_mask = None
            
            enc_src = encoder(src, src_mask)
            output = decoder(trg, enc_src, trg_mask, src_mask)
            
            predictions.append(output.item())
    
    return predictions

# Evaluate the model
predictions = evaluate_model(encoder, decoder, test_data)

# Denormalize predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plotting predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df.index[train_size+1:], data[train_size+1:], label='Actual')
plt.plot(df.index[train_size+1:], predictions, label='Predicted')
plt.title('Stock Price Prediction using Transformer')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
```

Trong ví dụ này:

- Chúng ta sử dụng mô hình Transformer Encoder-Decoder để dự đoán giá cổ phiếu dựa trên dữ liệu lịch sử.
- Dữ liệu được chuẩn bị và chia thành tập huấn luyện và tập kiểm tra.
- Mô hình được huấn luyện trên tập dữ liệu huấn luyện và đánh giá trên tập dữ liệu kiểm tra sử dụng optimizer Adam và hàm mất mát MSE.
- Cuối cùng, chúng ta đánh giá kết quả và vẽ biểu đồ so sánh giá thực và giá dự đoán.

Ví dụ này minh họa cách sử dụng mô hình Transformer (Encoder-Decoder) cho bài toán prediction trên dữ liệu chuỗi thời gian (time series), nhưng nó có thể được áp dụng tương tự cho các loại dữ liệu khác như dự báo thời tiết, dự đoán ngày nghỉ lễ, hoặc bất kỳ bài toán dự đoán nào khác có cấu trúc tương tự.

Transformer Encoder-Decoder là một kiến trúc mạng nơ-ron sử dụng trong Deep Learning, đặc biệt là hiệu quả trong các bài toán xử lý ngôn ngữ tự nhiên và dự đoán chuỗi. Kiến trúc này bao gồm hai phần chính: Transformer Encoder và Transformer Decoder, mỗi phần có các lớp Transformer được sử dụng để biểu diễn và xử lý thông tin từ đầu vào đến đầu ra.

### Transformer Encoder

Transformer Encoder nhận đầu vào là một chuỗi (sequence) và biểu diễn nó thành một dạng biểu diễn có ý nghĩa (semantic representation) bằng cách áp dụng một loạt các lớp Transformer.

1. **Embedding Layer**: Đầu tiên, các từ hoặc token trong chuỗi được biểu diễn bằng các vector embedding. Vector embedding này thường được học từ dữ liệu và có vai trò làm đại diện cho từ hoặc token trong không gian nhiều chiều.

2. **Positional Encoding**: Vì Transformer không sử dụng bất kỳ thứ tự nào của phần tử trong chuỗi, nên positional encoding (PE) được sử dụng để bổ sung thông tin về vị trí tuyệt đối của các phần tử trong chuỗi. Các positional encoding thêm các vector có giá trị cố định vào các vector embedding của các phần tử trong chuỗi.

3. **Multi-head Self-Attention Layer**: Đây là lớp chính trong Transformer Encoder. Mỗi vector embedding đi qua một quá trình self-attention, trong đó mỗi từ hoặc token "nhìn" vào tất cả các từ hoặc token khác trong cùng một chuỗi để tính toán mức độ quan trọng của chúng đối với từ hoặc token hiện tại. Multi-head self-attention cho phép mô hình học được nhiều cách biểu diễn khác nhau cho một từ hoặc token dựa trên các quan hệ khác nhau giữa các từ trong chuỗi.

4. **Feedforward Neural Network**: Sau khi đi qua các lớp self-attention, biểu diễn kết hợp của các từ sau đó được truyền qua một mạng nơ-ron truyền thẳng (feedforward neural network). Mạng nơ-ron này sẽ áp dụng một số lớp fully-connected với hàm kích hoạt để tạo ra biểu diễn cuối cùng của từng từ trong chuỗi.

5. **Layer Normalization và Residual Connection**: Mỗi lớp trong Transformer Encoder được bao gồm một layer normalization và một residual connection, giúp giảm thiểu vấn đề biến mất gradient và cải thiện khả năng huấn luyện của mô hình.

### Transformer Decoder

Sau khi Transformer Encoder đã biểu diễn được thông tin từ chuỗi đầu vào, Transformer Decoder tiếp tục xử lý để tạo ra chuỗi đầu ra (output sequence). Kiến trúc này thường được sử dụng trong các bài toán như dịch máy hoặc dự đoán chuỗi.

1. **Embedding Layer và Positional Encoding**: Tương tự như Transformer Encoder, đầu tiên, các token trong chuỗi đầu ra cũng được biểu diễn bằng vector embedding và positional encoding.

2. **Multi-head Self-Attention Layer (Masked)**: Khác với Transformer Encoder, trong quá trình giải mã (decoding), mỗi từ trong chuỗi đầu ra chỉ "nhìn" vào những từ đã được giải mã trước đó và không được "nhìn" vào tương lai. Điều này được thực hiện bằng cách áp dụng một mask để che dấu phần tương lai của chuỗi.

3. **Encoder-Decoder Attention Layer**: Sau khi qua lớp self-attention, mỗi từ trong chuỗi đầu ra sẽ nhận thông tin từ biểu diễn kết hợp của chuỗi đầu vào (chuỗi đã được mã hóa) thông qua một lớp attention. Lớp này giúp mô hình tập trung vào các phần quan trọng của chuỗi đầu vào để tạo ra chuỗi đầu ra phù hợp.

4. **Feedforward Neural Network**: Tương tự như Transformer Encoder, mỗi biểu diễn kết hợp sau đó được truyền qua một mạng nơ-ron truyền thẳng (feedforward neural network) để tạo ra biểu diễn cuối cùng của từng từ trong chuỗi đầu ra.

5. **Layer Normalization và Residual Connection**: Mỗi lớp trong Transformer Decoder cũng áp dụng layer normalization và residual connection như trong Transformer Encoder để giúp mô hình học tốt hơn và ổn định hơn.

### Tóm tắt

Transformer Encoder-Decoder là một kiến trúc mạng nơ-ron sử dụng nhiều lớp Transformer để biểu diễn và xử lý thông tin từ đầu vào đến đầu ra. Đây là một trong những kiến trúc hiệu quả nhất trong các mô hình sử dụng cho các bài toán dự đoán chuỗi như dịch máy, dự đoán chuỗi thời gian, hay bất kỳ bài toán nào đòi hỏi xử lý chuỗi dữ liệu có cấu trúc tương tự.


Để mô phỏng kiến trúc Transformer Encoder-Decoder sử dụng PyTorch, chúng ta sẽ tạo các lớp cơ bản của mô hình này bao gồm: Embedding Layer, Positional Encoding, Multi-head Self-Attention Layer, Feedforward Neural Network, và các lớp Layer Normalization và Residual Connection.

Ở đây, tôi sẽ giới hạn ví dụ với một số lớp cơ bản để bạn có thể hiểu cách các thành phần của Transformer hoạt động cùng nhau. Để đơn giản hóa, chúng ta sẽ sử dụng một số giá trị mặc định cho các tham số và không đi vào chi tiết về các cơ chế như attention mask.

### Định nghĩa các lớp cơ bản của Transformer Encoder-Decoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear transformation
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        
        # Apply attention to value
        att = torch.matmul(scores, V)
        
        # Concatenate heads and apply final linear transformation
        att = att.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        att = self.fc_o(att)
        
        return att

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(x + attn_output)
        ff_output = self.ff(out1)
        ff_output = self.dropout(ff_output)
        out2 = self.norm2(out1 + ff_output)
        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, enc_output):
        attn_output_1 = self.self_attn(x, x, x)
        attn_output_1 = self.dropout(attn_output_1)
        out1 = self.norm1(x + attn_output_1)
        
        attn_output_2 = self.enc_attn(out1, enc_output, enc_output)
        attn_output_2 = self.dropout(attn_output_2)
        out2 = self.norm2(out1 + attn_output_2)
        
        ff_output = self.ff(out2)
        ff_output = self.dropout(ff_output)
        out3 = self.norm3(out2 + ff_output)
        return out3

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x, enc_output):
        for layer in self.layers:
            x = layer(x, enc_output)
        return x

# Example usage
if __name__ == "__main__":
    # Define parameters
    num_layers = 3
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # Create encoder and decoder
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff)
    
    # Example input
    input_seq = torch.randn(10, 32, d_model)  # (sequence length, batch size, embedding size)
    
    # Encode
    enc_output = encoder(input_seq)
    
    # Decode
    output_seq = torch.randn(10, 32, d_model)  # (sequence length, batch size, embedding size)
    dec_output = decoder(output_seq, enc_output)
    
    print("Encoder output shape:", enc_output.shape)
    print("Decoder output shape:", dec_output.shape)
```

### Giải thích

1. **PositionalEncoding**: Lớp này thêm thông tin về vị trí của các phần tử trong chuỗi bằng cách sử dụng positional encoding, giúp mô hình biểu diễn được thứ tự của các phần tử trong chuỗi.
   
2. **MultiHeadAttention**: Lớp này thực hiện tính toán multi-head self-attention, trong đó mỗi từ trong chuỗi "nhìn" vào tất cả các từ khác trong cùng chuỗi để tính toán mức độ quan trọng của chúng đối với từ hiện tại.

3. **FeedForward**: Lớp này thực hiện một mạng nơ-ron truyền thẳng với hai lớp fully-connected và hàm kích hoạt ReLU, giúp biến đổi và kết hợp thông tin từ các lớp trước đó.

4. **EncoderLayer và DecoderLayer**: Lớp này xếp chồng các lớp như self-attention, layer normalization và feedforward để tạo thành một lớp hoàn chỉnh trong encoder và decoder.

5. **TransformerEncoder và TransformerDecoder**: Lớp này kết hợp nhiều lớp EncoderLayer hoặc DecoderLayer để xây dựng các phần encoder và decoder của mô hình Transformer.

### Tóm tắt



Trên đây là một mô phỏng đơn giản về kiến trúc Transformer Encoder-Decoder sử dụng PyTorch. Trong thực tế, các mô hình Transformer thường có thêm các chi tiết như attention mask, layer normalization, residual connection và các cơ chế khác để cải thiện hiệu suất và khả năng học của mô hình. Tuy nhiên, ví dụ trên cung cấp một cách nhìn tổng quan về cách các thành phần chính của Transformer hoạt động cùng nhau để xử lý dữ liệu chuỗi.




Hết.

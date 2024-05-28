---
title: '[Note] Hiểu hơn về KAN và cách triển khai lên pytorch'
date: 2024-05-28
permalink: /posts/2024/05/28/hieu-hon-ve-KAN-va-cach-trien-khai-len-pytorch/
tags:
  - KAN
  - Pytorch
--- 

1. MLP

```python
## dang viet o day (26/05/2024)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical



root = '/content/drive/MyDrive/[2024 - daily progress]/[2024-05-21] KAN models/'
dataset_N_baIoT = root + 'N-BaIoT/org/'
processed_file_data_N_baIOT = dataset_N_baIoT + 'data_train_and_test/' + 'data_processed_01.csv'

df = pd.read_csv(processed_file_data_N_baIOT)


# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :-1].values.astype(float))
        label = torch.tensor(self.data.iloc[idx, -1].astype(int))  # Remove .values

        if self.transform:
            features = self.transform(features)

        return features, label

# Define hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10

# normalize

features = df.columns[:-1]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

data = df

# Split into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)  # Adjust test_size if needed

# Create datasets
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the model
input_size = len(df.columns) - 1  # Number of features
hidden_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size, hidden_size, 11)
model.to(device)

from torchsummary import summary
summary(model, (input_size,))

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

import time
start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss = 0
    train_accuracy = 0
    with tqdm(train_loader) as pbar:
        for i, (features, labels) in enumerate(pbar):
            features = features.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (output.argmax(dim=1) == labels).float().mean().item()

            pbar.set_postfix(loss=loss.item(), accuracy=train_accuracy / (i + 1), lr=optimizer.param_groups[0]['lr'])

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device).float()
            labels = labels.to(device)

            output = model(features)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

```

2. DL (CNN, LSTM and GRU)

CNN model

```python
input_1 = Input (X_train.shape[1:],name='Inputlayer')

x = Conv1D(64, kernel_size=3, padding = 'same')(input_1)
x = MaxPool1D(3, strides = 2, padding = 'same')(x)

x = Conv1D(128, 3,strides=2, padding='same')(x)
x = Conv1D(128, 3,strides=2, padding='same')(x)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = Conv1D(128, 3,strides=2, padding='same')(x)
x = Conv1D(128, 3,strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(11, activation='softmax')(x)

model_cnn = Model(inputs=input_1, outputs=output_layer)
model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.summary()

callbacks = [EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    )
]
history=model_cnn.fit(X_train, y_train,epochs=75,batch_size=512,validation_data=(X_train, y_train),callbacks=callbacks,verbose=1)

```

GRU model

```python
input_1 = Input (X_train.shape[1:],name='Inputlayer')

x = GRU(128, return_sequences=True,activation='relu')(input_1)
x = BatchNormalization()(x)
x = GRU(64, return_sequences=True,activation='relu')(x)
x = BatchNormalization()(x)
x = GRU(32)(x)
output_layer = Dense(11, activation='softmax')(x)
model_gru = Model(inputs=input_1, outputs=output_layer)
model_gru.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model_gru.summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  min_delta=0.001,patience=3,
                                             mode='min',baseline=None, restore_best_weights=True )
history_gru=model_gru.fit(X_train, y_train,epochs=2,validation_split=0.2,callbacks=[stop_early],verbose=1)
```

LSTM

```python
input_1 = Input (X_train.shape[1:],name='Inputlayer')

x = LSTM(16, return_sequences=True,activation='relu')(input_1)
x = BatchNormalization()(x)
x = LSTM(32, return_sequences=True,activation='relu')(x)
x = Dropout(0.1)(x)
x = LSTM(16, return_sequences=True,activation='relu')(x)
x = BatchNormalization()(x)
x = LSTM(32)(x)
output_layer = Dense(11, activation='softmax')(x)

model_lstm = Model(inputs=input_1, outputs=output_layer)
model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  min_delta=0.001,patience=3,
                                             mode='min',baseline=None, restore_best_weights=True )
history_lstm=model_lstm.fit(X_train, y_train,epochs=50,batch_size=512,validation_data=(X_train, y_train),callbacks=[stop_early],verbose=1)

```

References
======

[Link tham khảo 01](https://drive.google.com/drive/folders/1yefyjgpUYjwyIrf4CWyeXtZZblY1XsIj)


Hết.

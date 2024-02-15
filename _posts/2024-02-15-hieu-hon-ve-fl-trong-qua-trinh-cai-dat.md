---
title: '[Note] Hiểu hơn về FL trong quá trình cài đặt'
date: 2024-02-15
permalink: /posts/2024/02/15/hieu-hon-ve-fl-trong-qua-trinh-cai-dat/
tags:
  - research
  - proposal
  - federated learning
--- 

Federated Learning
======

## Một số ghi chú khi cài đặt FL (Federated Learning)

Một số ghi chú khi cài đặt FL (Federated Learning)

### Load dataset

```python
def ReadData(dev_id):
  temp = pd.read_csv('{}/{}.{}.csv'.format(dataset_loc, dev_id, 'benign'))
  temp['class'] = 'benign'
  df = temp

  for i in range(0, len(classes)):
    temp = pd.read_csv('{}/{}.{}.csv'.format(dataset_loc, dev_id, classes[i]))
    temp['class'] = classes[i]
    df = df.append(temp)

  return df
def data_process(X, Y):
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  
  x_train = np.array(x_train)
  x_test = np.array(x_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  #Scale data
  t = MinMaxScaler()
  t.fit(x_train)
  x_train = t.transform(x_train)
  x_test = t.transform(x_test)

  return x_train, x_test, y_train, y_test

df1 = df1.reset_index(drop=True)
#get data from dataframe df1
X = df1.drop(columns=['class'])
Y = pd.get_dummies(df1['class'])
x_train, x_test, y_train, y_test = data_process(X, Y)
```

### Create client

```python
def create_client(x_train, y_train, num_clients = 10, initial = 'clients'):
  client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
  data = list(zip(x_train, y_train))
  random.shuffle(data)
  size = len(data) // num_clients
  shards = [data[i: i + size] for i in range(0, size * num_clients, size)]
  assert(len(shards) == len(client_names))
  return {client_names[i] : shards[i] for i in range(len(client_names))}

clients = create_client(x_train, y_train, num_clients = 10, initial='client')

```

### Một số hàm bổ sung

```python
def batch_data(data_shard, bs=32):
  data, label = zip(*data_shard)
  dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
  return dataset.shuffle(len(label)).batch(bs)


clients_batched = dict()
for (client_name, data) in clients.items():
  clients_batched[client_name] = batch_data(data)

test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

```

### Simple model

```python
class SimpleModel:
  @staticmethod
  def build(input_dim, add_hidden_layers, hidden_layer_size, num_outputs):
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation="tanh", input_shape=(input_dim,)))
    for i in range(add_hidden_layers):
        model.add(Dense(hidden_layer_size, activation="tanh"))
    model.add(Dense(num_outputs))
    model.add(Activation('softmax'))
    return model
  @staticmethod
  def build_lstm(input_dim, add_hidden_layers, hidden_layer_size, num_outputs):
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_dim, 1)))
    model.add(Dropout(0.5))
    for i in range(add_hidden_layers):
        model.add(Dense(hidden_layer_size, activation="tanh"))
    model.add(Dense(num_outputs))
    model.add(Activation('softmax'))
    return model
  @staticmethod
  def build_gru(input_dim, add_hidden_layers, hidden_layer_size, num_outputs):
    model = Sequential()
    model.add(GRU(128, input_shape=(input_dim, 1)))
    model.add(Dropout(0.5))
    for i in range(add_hidden_layers):
        model.add(Dense(hidden_layer_size, activation="tanh"))
    model.add(Dense(num_outputs))
    model.add(Activation('softmax'))
    return model
  
```

### Update functions

```python
def weight_scalling_factor(clients_trn_data, client_name):
  client_names = list(clients_trn_data.keys())
  bs = list(clients_trn_data[client_name])[0][0].shape[0]
  global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
  local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
  return local_count / global_count
```

```python
def scale_model_weights(weight, scalar):
  weight_final = []
  steps = len(weight)
  for i in range(steps):
    weight_final.append(scalar * weight[i])
  return weight_final
```

```python
def sum_scaled_weights(scaled_weight_list):
  avg_grad = list()
  for grad_list_tuple in zip(*scaled_weight_list):
    layer_mean = tf.math.reduce_sum(grad_list_tuple, axis = 0)
    avg_grad.append(layer_mean)
  return avg_grad
```

```python
def test_model(x_test1, y_test1, model, comm_round):
  cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  logits = model.predict(x_test1)
  loss = cce(y_test1, logits)
  acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(y_test1, axis=1))
  print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
  print('end test_model')
  return acc, loss
```

### Main functions

```python
# parameter
lr = 0.01
comms_round = 30
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer='adam'
input_dim = x_train.shape[1]

object_global = SimpleModel()
global_model = object_global.build_lstm(input_dim, 1, 128, len(classes) + 1)

for comm_round in range(comms_round):
  global_weights = global_model.get_weights()
  scaled_local_weight_list = list()
  client_names = list(clients_batched.keys())
  random.shuffle(client_names)
  print('comm_round: ', comm_round)
  for client in client_names:
    obj_local = SimpleModel()
    local_model = obj_local.build_lstm(input_dim, 1, 128, len(classes) + 1)
    local_model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    local_model.set_weights(global_weights)
    print('nameofclient: ', client)
    local_model.fit(clients_batched[client], epochs = 1, verbose = 0)

    scaling_factor = weight_scalling_factor(clients_batched, client)
    scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
    scaled_local_weight_list.append(scaled_weights)

    K.clear_session()
  
  average_weights = sum_scaled_weights(scaled_local_weight_list)
  global_model.set_weights(average_weights)

  test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
  for (x_test1, y_test1) in test_batched:
    global_acc, global_loss = test_model(x_test1, y_test1, global_model, comm_round)

```

### Sử dụng pretrained model vs cập nhật tham số với federated learning

Cơ bản về ý tưởng khi triển khai FL kết hợp với pretrained model
+ Thứ nhất, việc sử dụng mô hình tập trung bằng cách sử dụng vgg16 được thực hiện như sau:
Round 1, lúc này server sẽ khởi tạo tham số ban đầu, trường hợp vgg16 này sử dụng imagenet.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D

round = 1
center_model_path = root_path + "center/models/activity_%s.model" % round

baseModel = VGG16(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
model.save(center_model_path, save_format="h5")
```

+ Thứ hai, sau khi tính toán được trọng số của mỗi client, sẽ tính toán và cập nhật lại trọng số ở server (hiện tại tính toán giá trị trung bình)
Hàm tính trung bình tham số của hai client sẽ được thực hiện như dưới đây:
```python
import numpy as np

def average_params(weights_list, data_lens):
    """
    Returns the average of the params.
    """

    w_avg = (np.array(weights_list[0]) / data_lens[0])
    for i in range(1, len(weights_list)):
        w_avg += (np.array(weights_list[i]) / data_lens[i])
    return list(w_avg / len(weights_list))
```

Và quá trình cập nhật tham số như sau:
```python
from tensorflow import keras

round = 5 # Update ech round

client_1_model_path = root_path + "client-1/models/activity_%s.model" % (round - 1)
client_2_model_path = root_path + "client-2/models/activity_%s.model" % (round - 1)
center_model_path = root_path + "center/models/activity_%s.model" % round

model_1 = keras.models.load_model(client_1_model_path)
model_2 = keras.models.load_model(client_2_model_path)

params_1 = model_1.get_weights()
params_2 = model_2.get_weights()

avg_params = average_params([params_1, params_2])
model_1.set_weights(avg_params)

model_1.save(center_model_path, save_format="h5")

```

+ Thứ ba, đối với hình ảnh thì việc cần đó là phải bổ sung thêm ảnh cho dồi dào tập dữ liệu, và việc này có thể được thực hiện như sau:

```python
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

```

Và việc load mô hình sẽ được thực hiện như sau:

```python
# Load model from center
center_model_path = root_path + "center/models/activity_%s.model" % round
model = keras.models.load_model(center_model_path)

# compile model
print("[INFO] compiling model...")
opt = SGD(learning_rate=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train model
print("[INFO] training head...")
H = model.fit(
    x=trainAug.flow(X, y, batch_size=32),
    steps_per_epoch=len(X) // 32,
    epochs=arg_epochs)

full_train_loss.extend(H.history["loss"])
full_train_acc.extend(H.history["accuracy"])
```


Link ref: 

[#Federated kmeans](https://github.com/ourownstory/federated_kmeans)

[clustered-federated-learning](https://github.com/felisat/clustered-federated-learning/blob/master/clustered_federated_learning.ipynb)

[Federated-Learning-through-Distance-Based-Clustering](https://github.com/phanirohith/Federated-Learning-through-Distance-Based-Clustering)

[FedCHAR](https://github.com/youpengl/FedCHAR)

[FederatedDBSCAN](https://github.com/codiceSpaghetti/FederatedDBSCAN)

[Clustered-FL-GA](https://github.com/sagnik106/Clustered-FL-GA)

Hết.

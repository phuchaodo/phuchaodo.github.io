---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng

Có nhiều thuật toán phổ biến trong lĩnh vực Học tăng cường (Reinforcement Learning), mỗi thuật toán có những ưu điểm và điểm yếu riêng. Dưới đây là một số thuật toán phổ biến trong lĩnh vực này:

1. **Q-Learning**: Là một trong những thuật toán học tăng cường đơn giản nhất và phổ biến nhất. Q-Learning tập trung vào việc học các hàm giá trị hành động (action-value function) để đưa ra quyết định hành động tối ưu.

2. **Deep Q-Networks (DQN)**: Kết hợp Q-Learning với mạng nơ-ron sâu (deep neural networks) để xử lý không gian trạng thái lớn hơn và các hàm giá trị hành động phức tạp hơn.

3. **Policy Gradient Methods**: Tập trung vào việc tối ưu hàm chính sách (policy function) trực tiếp, thay vì tối ưu hàm giá trị. Các phương pháp trong nhóm này bao gồm REINFORCE, Actor-Critic, và các biến thể như PPO (Proximal Policy Optimization) và A3C (Asynchronous Advantage Actor-Critic).

4. **Actor-Critic Methods**: Kết hợp cả phương pháp chính sách (policy) và hàm giá trị (value function), với một actor (đại diện cho chính sách) và một critic (đại diện cho hàm giá trị).

5. **Deep Deterministic Policy Gradient (DDPG)**: Một biến thể của Actor-Critic dành cho các bài toán hành động liên tục, sử dụng mạng nơ-ron sâu để học chính sách và hàm giá trị.

6. **Twin Delayed DDPG (TD3)**: Một biến thể cải tiến của DDPG, nhắm vào việc giảm thiểu các sai số trong hàm giá trị.

7. **Soft Actor-Critic (SAC)**: Một phương pháp Actor-Critic khác, nhằm tối ưu hóa chính sách với một phương pháp tối ưu hóa mềm (soft optimization).

8. **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**: Được áp dụng cho các bài toán với nhiều tác nhân học tăng cường, MADDPG là một phương pháp mở rộng của DDPG cho môi trường đa tác nhân.

Đây chỉ là một số ví dụ tiêu biểu và phổ biến. Các thuật toán này thường được áp dụng và điều chỉnh để phù hợp với từng bài toán cụ thể trong lĩnh vực học tăng cường.


Q-Learning là một thuật toán học tăng cường cơ bản trong đó chúng ta học một hàm giá trị hành động (action-value function) Q từ việc tương tác với môi trường. Chúng ta cần một bảng giá trị Q để lưu trữ các giá trị ước tính cho từng cặp trạng thái-hành động. Thuật toán này dựa trên việc cập nhật giá trị Q bằng cách sử dụng phương pháp lặp mở rộng (Bellman equation).

Dưới đây là một ví dụ minh họa về cách triển khai Q-Learning bằng Python và PyTorch. Chúng ta sẽ sử dụng một môi trường đơn giản là Grid World (thế giới lưới) với các trạng thái và hành động đơn giản để minh họa.

Đầu tiên, chúng ta cần import các thư viện cần thiết:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

Tiếp theo, ta sẽ định nghĩa lớp cho mô hình Q-Learning, trong đó bao gồm một bảng giá trị Q và các phương thức để cập nhật giá trị Q và lựa chọn hành động dựa trên chính sách epsilon-greedy.

```python
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q table
        self.Q = torch.zeros(num_states, num_actions)
        
    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # explore
        else:
            with torch.no_grad():
                return torch.argmax(self.Q[state]).item()  # exploit
    
    def update_q(self, state, action, reward, next_state):
        # Q-Learning update rule
        best_next_action = torch.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error
```

Ở đây:
- `num_states` là số lượng trạng thái trong môi trường.
- `num_actions` là số lượng hành động có thể thực hiện từ mỗi trạng thái.
- `learning_rate` là tỷ lệ học (learning rate) cho việc cập nhật giá trị Q.
- `discount_factor` là hệ số chiết khấu (discount factor) trong phương trình Bellman.
- `epsilon` là giá trị cho chiến lược epsilon-greedy, để thúc đẩy khám phá.

Tiếp theo, chúng ta sẽ cài đặt một ví dụ về việc sử dụng Q-Learning để giải quyết bài toán Grid World đơn giản:

```python
# Define the environment (Grid World)
num_states = 16  # Number of states
num_actions = 4  # Number of actions (up, down, left, right)

# Create Q-Learning agent
agent = QLearningAgent(num_states, num_actions)

# Training loop
num_episodes = 1000
max_steps_per_episode = 100

for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Start state randomly
    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        # Perform action and observe next_state, reward
        next_state = transition_function(state, action)  # hypothetical transition function
        reward = reward_function(state, action, next_state)  # hypothetical reward function
        # Update Q value
        agent.update_q(state, action, reward, next_state)
        state = next_state
```

Trong đoạn mã trên:
- `transition_function` và `reward_function` là hai hàm giả định để thực hiện bước chuyển trạng thái và tính toán phần thưởng dựa trên hành động được chọn.
- Mỗi episode bắt đầu với một trạng thái ngẫu nhiên, và agent sử dụng Q-Learning để tối ưu hóa bảng giá trị Q thông qua các bước lặp.

Đây là một bản triển khai đơn giản của Q-Learning trong Python và PyTorch. Bạn có thể điều chỉnh và mở rộng mã này cho các môi trường và bài toán khác nhau trong lĩnh vực học tăng cường.


Deep Q-Networks (DQN) là một phương pháp tiên tiến trong lĩnh vực học tăng cường, kết hợp giữa thuật toán Q-Learning và mạng nơ-ron sâu để giải quyết các bài toán có không gian trạng thái lớn và phức tạp hơn. DQN đã giải quyết được các vấn đề của Q-Learning khi mà Q-table truyền thống không thể tổng hợp các bảng giá trị Q cho các môi trường có không gian trạng thái lớn.

### Chi tiết về Deep Q-Networks (DQN)

1. **Hàm giá trị hành động (Action-Value Function)**:
   DQN học một hàm giá trị hành động \( Q(s, a; \theta) \), trong đó \( s \) là trạng thái, \( a \) là hành động, và \( \theta \) là tham số của mạng nơ-ron.

2. **Cập nhật hàm giá trị hành động**:
   DQN sử dụng phương pháp lặp mở rộng (Bellman equation) để cập nhật giá trị Q:
   \[
   Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)
   \]
   Trong đó:
   - \( \alpha \) là tỷ lệ học (learning rate).
   - \( r \) là phần thưởng nhận được khi thực hiện hành động \( a \) từ trạng thái \( s \).
   - \( \gamma \) là hệ số chiết khấu (discount factor).
   - \( s' \) là trạng thái kế tiếp sau khi thực hiện hành động \( a \).
   - \( \theta^- \) là các tham số của mạng nơ-ron cũ (target network) được cập nhật một cách chậm hơn so với mạng nơ-ron chính (policy network).

3. **Mạng nơ-ron chính (Policy Network)**:
   DQN sử dụng một mạng nơ-ron sâu để ước tính hàm giá trị hành động \( Q(s, a; \theta) \). Thông thường, mạng nơ-ron này bao gồm các lớp convolutional (cho các môi trường hình ảnh) và các lớp fully connected để xử lý dữ liệu đầu vào và đưa ra các dự đoán về giá trị Q cho mỗi hành động.

4. **Target Network**:
   Để ổn định quá trình học, DQN sử dụng một mạng nơ-ron cũ (target network) để tính toán giá trị mục tiêu \( r + \gamma \max_{a'} Q(s', a'; \theta^-) \). Target network được cập nhật từ policy network mỗi một vài bước để giảm thiểu vấn đề động chạm giữa các tham số khi cập nhật.

### Triển khai DQN bằng Python và PyTorch

Dưới đây là một ví dụ về cách triển khai DQN sử dụng PyTorch cho bài toán môi trường học tăng cường đơn giản như CartPole:

```python
import gym
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.99, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()
    
    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### Giải thích code:

- `QNetwork`: Định nghĩa mạng nơ-ron cho hàm giá trị Q với các lớp fully connected. Hàm kích hoạt sử dụng là ReLU cho các lớp ẩn và không có hàm kích hoạt ở lớp đầu ra vì đây là bài toán hồi quy.
  
- `DQNAgent`: Lớp định nghĩa cho agent DQN với các phương thức như `memorize` (lưu trữ trạng thái hành động), `select_action` (chọn hành động dựa trên chính sách epsilon-greedy), `experience_replay` (lặp lại trải nghiệm), và `update_target_network` (cập nhật mạng mục tiêu).

- `experience_replay`: Hàm thực hiện quá trình lặp lại trải nghiệm. Nó lấy một mẫu ngẫu nhiên từ bộ nhớ và thực hiện cập nhật mạng theo phương pháp lặp mở rộng.

- `update_target_network`: Hàm cập nhật mạng nơ-ron mục tiêu bằng cách sao chép tham số từ mạng chính.

### Huấn luyện Agent

```python
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

num_episodes = 1000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent

.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.memorize(state, action, reward, next_state, done)
        agent.experience_replay()
        agent.update_target_network()
        
        total_reward += reward
        state = next_state
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon}")
```

Trong đoạn mã huấn luyện:
- Chúng ta sử dụng môi trường `CartPole-v1` từ thư viện Gym để huấn luyện agent.
- Mỗi episode, agent tương tác với môi trường, lưu trữ trải nghiệm, thực hiện lặp lại trải nghiệm và cập nhật mạng nơ-ron mục tiêu.
- Giá trị epsilon giảm dần theo thời gian để giảm dần khám phá và tăng khai phá khi huấn luyện đi sâu hơn.

Đây là một ví dụ đơn giản về việc triển khai Deep Q-Networks (DQN) bằng Python và PyTorch. Bạn có thể điều chỉnh và mở rộng mã này cho các môi trường và bài toán phức tạp hơn trong lĩnh vực học tăng cường.


Policy Gradient Methods là một lớp các thuật toán trong học tăng cường tập trung vào việc tối ưu hóa trực tiếp chính sách (policy) mà không cần ước tính giá trị hành động (action-value function) như Q-Learning hay DQN. Các thuật toán này thường áp dụng trong các bài toán có không gian hành động liên tục và phức tạp hơn. Dưới đây, chúng ta sẽ đi vào chi tiết và triển khai một ví dụ đơn giản về Policy Gradient Method sử dụng PyTorch.

### Chi tiết về Policy Gradient Methods

1. **Chính sách (Policy)**:
   Chính sách là một hàm \( \pi(a|s; \theta) \), cho biết xác suất lựa chọn hành động \( a \) khi ở trạng thái \( s \), được tham số hóa bởi \( \theta \).

2. **Mục tiêu tối ưu**:
   Chúng ta tối đa hóa hàm tổng thưởng mong đợi (expected return):
   \[
   J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^T \gamma^t r_t \right]
   \]
   Trong đó \( \tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots) \) là một quỹ đạo (trajectory), và \( \gamma \) là hệ số chiết khấu.

3. **Gradient của chính sách**:
   Gradient của hàm mục tiêu \( J(\theta) \) có thể được tính bằng cách sử dụng kỹ thuật REINFORCE:
   \[
   \nabla_{\theta} J(\theta) \approx \frac{1}{|\tau|} \sum_{t=0}^T \nabla_{\theta} \log \pi(a_t|s_t; \theta) \left( \sum_{t'=t}^T \gamma^{t' - t} r_{t'} \right)
   \]

### Triển khai Policy Gradient Method bằng Python và PyTorch

Dưới đây là một ví dụ về triển khai thuật toán Policy Gradient bằng cách sử dụng một mạng nơ-ron đơn giản cho chính sách và môi trường Gym (CartPole):

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

# Define Policy Gradient Agent
class PolicyGradientAgent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, gamma=0.99):
        self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

# Training loop
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 128

agent = PolicyGradientAgent(input_size, hidden_size, output_size)

num_episodes = 1000
max_steps_per_episode = 500
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    
    for step in range(max_steps_per_episode):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        if done:
            agent.update_policy(rewards, log_probs)
            break
        
        state = next_state
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

env.close()
```

### Giải thích code:

- `PolicyNetwork`: Là một mạng nơ-ron đơn giản với hai lớp fully connected. Đầu ra của lớp cuối cùng là xác suất của các hành động được tính bằng softmax để đảm bảo tổng xác suất bằng 1.

- `PolicyGradientAgent`: Lớp đại diện cho agent Policy Gradient với các phương thức như `select_action` (lựa chọn hành động dựa trên chính sách), và `update_policy` (cập nhật chính sách dựa trên gradient).

- Trong vòng lặp huấn luyện, agent tương tác với môi trường, thu thập phần thưởng và log-probability của các hành động, sau đó cập nhật chính sách bằng cách gọi `update_policy` sau khi một episode kết thúc.

Đây là một ví dụ đơn giản về triển khai Policy Gradient Method bằng Python và PyTorch. Bạn có thể điều chỉnh và mở rộng mã này cho các môi trường và bài toán phức tạp hơn trong lĩnh vực học tăng cường.


Actor-Critic Methods là một lớp các thuật toán trong học tăng cường kết hợp cả hai mô hình Actor (đại diện cho chính sách) và Critic (đại diện cho hàm giá trị hành động) để cải thiện quá trình học. Mục tiêu của Actor-Critic là tối ưu hóa trực tiếp chính sách (Actor) và học một hàm giá trị hành động (Critic) để đánh giá chất lượng của chính sách hiện tại.

### Chi tiết về Actor-Critic Methods

1. **Actor (Policy Network)**:
   - Một mô hình mạng nơ-ron, thường là một mạng nơ-ron sâu, đưa ra hành động dựa trên trạng thái hiện tại.
   - Được tham số hóa bởi \( \theta^{\pi} \).
   - Đầu ra là một phân phối xác suất của các hành động.

2. **Critic (Value Network)**:
   - Một mô hình mạng nơ-ron, cũng thường là một mạng nơ-ron sâu, ước tính hàm giá trị hành động (action-value function).
   - Được tham số hóa bởi \( \theta^{V} \).
   - Đầu ra là giá trị ước tính của hàm giá trị hành động \( V(s) \).

3. **Mục tiêu tối ưu**:
   - Tối đa hóa giá trị hành động thông qua hàm giá trị hành động: \( J(\theta^{\pi}) = \mathbb{E}_{\tau \sim \pi_{\theta^{\pi}}} \left[ \sum_{t=0}^T \gamma^t r_t \right] \).
   - Học một hàm giá trị hành động để ước tính giá trị của chính sách hiện tại: \( V^{\pi}(s) \).

4. **Gradient của Actor và Critic**:
   - Gradient của Actor (Policy Gradient): \( \nabla_{\theta^{\pi}} J(\theta^{\pi}) \approx \mathbb{E}_{\tau \sim \pi_{\theta^{\pi}}} \left[ \sum_{t=0}^T \nabla_{\theta^{\pi}} \log \pi(a_t|s_t; \theta^{\pi}) \cdot A^{\pi}(s_t, a_t) \right] \), với \( A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t) \) là lợi ích hành động.
   - Gradient của Critic: Được tính bằng cách giải quyết bài toán hồi quy bình phương (least squares regression) với mục tiêu là giá trị trả về \( R_t \).

### Triển khai Actor-Critic Method bằng Python và PyTorch

Dưới đây là một ví dụ về triển khai Actor-Critic Method bằng PyTorch trong môi trường học tăng cường đơn giản như CartPole:

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, input_size, hidden_size, output_size, actor_lr=0.01, critic_lr=0.01, gamma=0.99):
        self.actor = Actor(input_size, hidden_size, output_size)
        self.critic = Critic(input_size, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def update(self, rewards, log_probs, values):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        advantage = returns - values
        
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

# Training loop
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 128

agent = ActorCriticAgent(input_size, hidden_size, output_size)

num_episodes = 1000
max_steps_per_episode = 500

for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    values = []
    
    for step in range(max_steps_per_episode):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = torch.from_numpy(state).float().unsqueeze(0)
        value = agent.critic(state)
        
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        
        state = next_state
        
        if done:
            agent.update(rewards, log_probs, values)
            break
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

env.close()
```

### Giải thích code:

- `Actor`: Là một mạng nơ-ron đơn giản với hai lớp fully connected. Đầu ra của lớp cuối cùng là phân phối xác suất của các hành động được tính bằng softmax.

- `Critic`: Là một mạng nơ-ron đơn giản với hai lớp fully connected. Đầu ra của lớp cuối cùng là giá trị ước tính của hàm giá trị hành động.

- `ActorCriticAgent`: Lớp đại diện cho agent Actor-Critic với các phương thức như `select_action` (lựa chọn hành động dựa trên chính sách), và `update` (cập nhật các tham số của Actor và Critic).

- Trong vòng lặp huấn luyện, agent tương tác với môi trường, thu thập phần thưởng, log-probability và giá trị từ mô hình Critic, sau đó cập nhật Actor và Critic bằng cách gọi phương thức `update` sau khi một episode kết thúc.

Đây là một ví dụ đơn giản về triển khai Actor-Critic Method bằng Python và PyTorch. Bạn có thể điều chỉnh và mở rộng mã này cho các môi trường và bài toán phức tạp hơn trong lĩnh vực học tăng cường.



Deep Deterministic Policy Gradient (DDPG) là một thuật toán học tăng cường phổ biến trong các bài toán với không gian hành động liên tục. Nó kết hợp các ý tưởng từ các phương pháp Actor-Critic và Q-learning để học một chính sách (policy) liên tục và hàm giá trị hành động (action-value function).

### Chi tiết về Deep Deterministic Policy Gradient (DDPG)

1. **Actor (Policy Network)**:
   - Một mạng nơ-ron sâu (deep neural network) đưa ra hành động dựa vào trạng thái hiện tại.
   - Được tham số hóa bởi \( \theta^{\mu} \).
   - Chúng ta sử dụng hàm tanh cho đầu ra của mô hình để đảm bảo rằng hành động được sinh ra nằm trong khoảng giới hạn của không gian hành động.

2. **Critic (Action-Value Network)**:
   - Một mạng nơ-ron sâu (deep neural network) để ước tính hàm giá trị hành động (action-value function).
   - Được tham số hóa bởi \( \theta^{Q} \).
   - Nhận đầu vào là trạng thái và hành động, đưa ra giá trị ước tính của hàm giá trị hành động \( Q(s, a) \).

3. **Mục tiêu tối ưu**:
   - Tối đa hóa giá trị hành động thông qua hàm giá trị hành động: \( J(\theta^{\mu}) = \mathbb{E}_{s \sim \rho(s)} \left[ \mathbb{E}_{a \sim \pi_{\theta^{\mu}}} [Q(s, a)] \right] \).
   - Học một hàm giá trị hành động để đánh giá chất lượng của chính sách hiện tại và sử dụng gradient của hàm giá trị để cập nhật chính sách.

4. **Gradient của Actor và Critic**:
   - Gradient của Actor (Policy Gradient): \( \nabla_{\theta^{\mu}} J(\theta^{\mu}) \approx \mathbb{E}_{s \sim \rho(s), a \sim \pi_{\theta^{\mu}}} \left[ \nabla_{\theta^{\mu}} \mu(s; \theta^{\mu}) \cdot \nabla_a Q(s, a| \theta^{Q})|_{a = \mu(s)} \right] \).
   - Gradient của Critic: Được tính bằng cách giải quyết bài toán hồi quy bình phương (least squares regression) với mục tiêu là giá trị trả về \( R_t \).

### Triển khai Deep Deterministic Policy Gradient (DDPG) bằng Python và PyTorch

Dưới đây là một ví dụ về triển khai DDPG bằng PyTorch trong môi trường học tăng cường đơn giản như Pendulum-v0:

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output range: [-1, 1]
        return x

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size + output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_size=256, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size, hidden_size, action_size)
        self.actor_target = Actor(state_size, hidden_size, action_size)
        self.critic = Critic(state_size + action_size, hidden_size, 1)
        self.critic_target = Critic(state_size + action_size, hidden_size, 1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        
        # Initialize target networks with the same weights as the original networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy().squeeze(0)
        action += noise * np.random.randn(self.action_size)
        return np.clip(action, -1.0, 1.0)
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Update Critic
        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        Q_next = self.critic_target(next_states, next_actions.detach())
        target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * Q_next
        critic_loss = F.mse_loss(Qvals, target_Q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        policy_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training loop
env = gym.make('Pendulum-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = DDPGAgent(state_size, action_size)

num_episodes = 1000
max_steps_per_episode = 500

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")

env.close()
```

### Giải thích code:

- `Actor`: Mạng nơ-ron với hai lớp fully connected và hàm tanh ở lớp cuối cùng để đảm bảo rằng đầu ra nằm trong khoảng [-1, 1], phù hợp với không gian hành động liên tục.

- `Critic`: Mạng nơ-ron với hai lớp fully connected, nhận đầu vào là trạng thái và hành động, đưa ra giá trị ước tính của hàm gi


Để trình bày chi tiết và cung cấp mã nguồn Python sử dụng các thuật toán học tăng cường đã liệt kê (Q-Learning, Deep Q-Network (DQN), và Proximal Policy Optimization (PPO)), chúng ta sẽ sử dụng thư viện PyTorch để triển khai.

### Q-Learning

Q-Learning là một thuật toán học tăng cường cơ bản để học hành vi tối ưu trong một môi trường.

```python
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        if not done:
            max_future_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_future_q
        else:
            target_q = reward
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
```

### Deep Q-Network (DQN)

DQN sử dụng mạng nơ-ron sâu để xấp xỉ hàm giá trị hành động.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

        q_value = self.q_network(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_q_value = self.q_network(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.functional.mse_loss(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
```

### Proximal Policy Optimization (PPO)

PPO là một thuật toán học sâu phổ biến, tối ưu hóa hàm mục tiêu bằng cách áp dụng các hạn chế về khoảng cách giữa các chính sách.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=0.2, clip_value=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_network = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy_network(state)
        action_probs = action_probs.cpu().numpy().flatten()
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def learn(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        _, critic_values = self.policy_network(states)
        _, next_critic_values = self.policy_network(next_states)

        td_targets = rewards + self.gamma * next_critic_values * (1 - dones)
        advantages = td_targets - critic_values

        for _ in range(10):  # PPO epoch
            action_probs, critic_values = self.policy_network(states)
            action_probs = action_probs.gather(1, actions)
            action_probs_next, _ = self.policy_network(next_states)
            action_probs_next = action_probs_next.gather(1, actions)

            ratio = action_probs / action_probs_next
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(td_targets.detach(), critic_values)

            self.optimizer.zero_grad()
            loss = actor_loss + critic_loss
            loss.backward()
            self.optimizer.step()
```

Trên đây là mã nguồn Python sử dụng PyTorch để triển khai ba thuật toán học tăng cường: Q-Learning, Deep Q-Network (DQN), và Proximal Policy Optimization (PPO). Mỗi thuật toán được triển khai trong một class riêng biệt, và mỗi class bao gồm các phương thức cho việc chọn hành động, học và cập nhật mạng nơ-ron. Lưu ý rằng đây là các phiên bản đơn giản và có thể cần điều chỉnh thêm để phù hợp với từng bài toán cụ thể.


Để minh họa ví dụ về phân bổ tài nguyên về năng lượng tiêu thụ trong mạng vệ tinh, chúng ta sẽ giả định có một mạng vệ tinh gồm nhiều vệ tinh cần phân bổ tài nguyên năng lượng. Mỗi vệ tinh có thể cần quyết định lượng năng lượng mà nó sẽ sử dụng để thực hiện các nhiệm vụ khác nhau.

### Dữ liệu mẫu

Giả sử chúng ta có 5 vệ tinh và cần phân bổ năng lượng tiêu thụ cho mỗi vệ tinh. Dữ liệu mẫu có thể được biểu diễn như sau:

```python
import numpy as np

# Số lượng vệ tinh và loại năng lượng cần phân bổ
num_satellites = 5
num_energy_types = 3  # Ví dụ: solar, battery, nuclear

# Dữ liệu mẫu về năng lượng tiêu thụ của từng loại cho từng vệ tinh
# Ví dụ: mỗi hàng là một vệ tinh, mỗi cột là một loại năng lượng
energy_consumption_data = np.array([
    [0.5, 0.3, 0.2],  # Satellite 1: solar=0.5, battery=0.3, nuclear=0.2
    [0.4, 0.4, 0.2],  # Satellite 2
    [0.6, 0.2, 0.2],  # Satellite 3
    [0.3, 0.5, 0.2],  # Satellite 4
    [0.7, 0.1, 0.2],  # Satellite 5
])

# Tổng năng lượng tiêu thụ của các loại năng lượng cho mỗi vệ tinh
total_energy_consumption = np.sum(energy_consumption_data, axis=1)

print("Energy consumption data (per satellite):")
for i in range(num_satellites):
    print(f"Satellite {i+1}: {energy_consumption_data[i]}")

print("\nTotal energy consumption for each satellite:")
for i in range(num_satellites):
    print(f"Satellite {i+1}: {total_energy_consumption[i]}")
```

Kết quả in ra màn hình sẽ là:

```
Energy consumption data (per satellite):
Satellite 1: [0.5 0.3 0.2]
Satellite 2: [0.4 0.4 0.2]
Satellite 3: [0.6 0.2 0.2]
Satellite 4: [0.3 0.5 0.2]
Satellite 5: [0.7 0.1 0.2]

Total energy consumption for each satellite:
Satellite 1: 1.0
Satellite 2: 1.0
Satellite 3: 1.0
Satellite 4: 1.0
Satellite 5: 1.0
```

### Sử dụng thuật toán học tăng cường (ví dụ Q-Learning) để phân bổ năng lượng

Dưới đây là một ví dụ cơ bản về việc sử dụng Q-Learning để phân bổ năng lượng cho các vệ tinh dựa trên tổng năng lượng tiêu thụ đã tính toán từ dữ liệu mẫu trên.

```python
class QLearningEnergyAllocation:
    def __init__(self, num_satellites, num_energy_types, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99):
        self.num_satellites = num_satellites
        self.num_energy_types = num_energy_types
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_satellites, num_energy_types))

    def choose_energy_type(self, satellite_idx):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_energy_types)
        else:
            return np.argmax(self.q_table[satellite_idx])

    def learn(self, satellite_idx, energy_type, reward):
        current_q = self.q_table[satellite_idx, energy_type]
        max_future_q = np.max(self.q_table[satellite_idx])
        target_q = reward + self.discount_factor * max_future_q
        self.q_table[satellite_idx, energy_type] += self.learning_rate * (target_q - current_q)

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

# Tạo đối tượng Q-Learning để phân bổ năng lượng
q_learning_agent = QLearningEnergyAllocation(num_satellites, num_energy_types)

# Huấn luyện và phân bổ năng lượng cho mỗi vệ tinh
num_episodes = 1000  # Số lượng lần huấn luyện
for episode in range(num_episodes):
    total_reward = 0
    for satellite_idx in range(num_satellites):
        chosen_energy_type = q_learning_agent.choose_energy_type(satellite_idx)
        reward = total_energy_consumption[satellite_idx]  # Reward được chọn là tổng năng lượng tiêu thụ của vệ tinh
        q_learning_agent.learn(satellite_idx, chosen_energy_type, reward)
        total_reward += reward

    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Total reward: {total_reward}")

# In ra bảng phân bổ năng lượng đã học được sau khi huấn luyện
print("\nFinal energy allocation (Q-table):")
for i in range(num_satellites):
    print(f"Satellite {i+1}: {q_learning_agent.q_table[i]}")
```

Trong ví dụ trên, chúng ta tạo một đối tượng `QLearningEnergyAllocation` để thực hiện việc phân bổ năng lượng cho từng vệ tinh sử dụng Q-Learning. Mỗi vệ tinh sẽ chọn loại năng lượng dựa trên chiến lược epsilon-greedy, và sau đó cập nhật Q-table dựa trên giá trị reward (tổng năng lượng tiêu thụ của vệ tinh). Sau khi huấn luyện xong, chúng ta sẽ in ra bảng phân bổ năng lượng đã học được.

Đây là một ví dụ đơn giản và thực tế có thể phức tạp hơn, tùy thuộc vào yêu cầu cụ thể của bài toán và dữ liệu thực tế của mạng vệ tinh.


Hãy sử dụng thuật toán Deep Q-Network (DQN) để giải quyết bài toán phân bổ tài nguyên năng lượng cho các vệ tinh trong mạng vệ tinh.

### Sử dụng Deep Q-Network (DQN)

DQN là một thuật toán học tăng cường sử dụng mạng nơ-ron sâu để xấp xỉ hàm giá trị hành động (action-value function). Chúng ta sẽ triển khai DQN để học cách phân bổ năng lượng cho các vệ tinh trong mạng vệ tinh, dựa trên dữ liệu về tổng năng lượng tiêu thụ của từng vệ tinh.

Đầu tiên, chúng ta cần import các thư viện cần thiết và định nghĩa lớp mạng nơ-ron sâu (DQN).

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Số lượng vệ tinh và loại năng lượng cần phân bổ
num_satellites = 5
num_energy_types = 3  # Ví dụ: solar, battery, nuclear

# Dữ liệu mẫu về năng lượng tiêu thụ của từng loại cho từng vệ tinh
energy_consumption_data = np.array([
    [0.5, 0.3, 0.2],  # Satellite 1: solar=0.5, battery=0.3, nuclear=0.2
    [0.4, 0.4, 0.2],  # Satellite 2
    [0.6, 0.2, 0.2],  # Satellite 3
    [0.3, 0.5, 0.2],  # Satellite 4
    [0.7, 0.1, 0.2],  # Satellite 5
])

# Tổng năng lượng tiêu thụ của các loại năng lượng cho mỗi vệ tinh
total_energy_consumption = np.sum(energy_consumption_data, axis=1)

# Chuyển đổi dữ liệu thành tensor PyTorch
energy_consumption_tensor = torch.tensor(energy_consumption_data, dtype=torch.float32)

# Định nghĩa lớp mạng nơ-ron sâu (DQN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Định nghĩa agent sử dụng DQN để phân bổ năng lượng
class DQNEnergyAllocationAgent:
    def __init__(self, num_satellites, num_energy_types, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
        self.num_satellites = num_satellites
        self.num_energy_types = num_energy_types
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dqn_network = DQN(num_energy_types, num_satellites).to(self.device)
        self.optimizer = optim.Adam(self.dqn_network.parameters(), lr=learning_rate)

    def choose_energy_distribution(self, energy_consumption_tensor):
        state = energy_consumption_tensor.flatten().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.dqn_network(state)
        return torch.argmax(q_values).item()

    def learn(self, energy_consumption_tensor):
        state = energy_consumption_tensor.flatten().unsqueeze(0).to(self.device)
        action = self.choose_energy_distribution(energy_consumption_tensor)

        reward = -torch.sum(energy_consumption_tensor[0, action])  # Phạt năng lượng tiêu thụ
        next_state = energy_consumption_tensor.flatten().unsqueeze(0).to(self.device)

        q_value = self.dqn_network(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_q_value = self.dqn_network(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + self.gamma * next_q_value

        loss = nn.functional.mse_loss(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

# Tạo đối tượng DQNEnergyAllocationAgent và huấn luyện
dqn_agent = DQNEnergyAllocationAgent(num_satellites, num_energy_types)

num_episodes = 1000
for episode in range(num_episodes):
    dqn_agent.learn(energy_consumption_tensor)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}")

# In ra bảng phân bổ năng lượng đã học được sau khi huấn luyện
print("\nFinal energy allocation:")
for i in range(num_satellites):
    energy_distribution = dqn_agent.choose_energy_distribution(energy_consumption_tensor)
    print(f"Satellite {i+1}: Energy type {energy_distribution}")
```

Trong ví dụ này, chúng ta đã triển khai DQN để học cách phân bổ năng lượng cho các vệ tinh trong mạng vệ tinh. Mỗi vệ tinh sẽ chọn một loại năng lượng để giảm thiểu tổng năng lượng tiêu thụ (được tính là phạt trong hàm reward). Quá trình huấn luyện sẽ cập nhật các trọng số của mạng DQN dựa trên lỗi (loss) giữa giá trị Q tính toán và giá trị Q mục tiêu.

Lưu ý rằng đây là một ví dụ cơ bản và có thể cần điều chỉnh thêm để phù hợp với yêu cầu cụ thể của bài toán. Trong thực tế, bạn có thể cần cải thiện mô hình DQN, tăng số lượng episodes, hay sử dụng các kỹ thuật như Double DQN, Dueling DQN để cải thiện hiệu suất và ổn định của thuật toán.


Hết.

import torch
import torch.nn as nn
from Environment import Environment
import random
from torch.optim.lr_scheduler import StepLR
# class DQN(nn.Module):
#     def __init__(self, history_dim, action_dim, hidden_dim=128):
#         super(DQN, self).__init__()

#         combined_dim = history_dim + action_dim

#         self.q_net = nn.Sequential(
#             nn.Linear(combined_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),  # New hidden layer
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)  # Output Q-value
#         )

#     def forward(self, history_vec, actions_vecs):
#         """
#         Args:
#             history_vec: Tensor (batch_size, history_dim)
#             actions_vecs: Tensor (batch_size, num_actions, action_dim)

#         Returns:
#             q_values: Tensor (batch_size, num_actions)
#         """
#         batch_size, num_actions, action_dim = actions_vecs.shape
#         history_dim = history_vec.shape[1]

#         history_repeated = history_vec.unsqueeze(1).repeat(1, num_actions, 1)
#         concat = torch.cat([history_repeated, actions_vecs], dim=-1)
#         concat_flat = concat.view(-1, history_dim + action_dim)

#         q_flat = self.q_net(concat_flat)
#         q_values = q_flat.view(batch_size, num_actions)

#         return q_values

# class DQN(nn.Module):
#     def __init__(self, input_dim, num_actions, hidden_dim=128, dropout_prob=0.2, n_heads=4):
#         super(DQN, self).__init__()

#         self.embedding = nn.Linear(input_dim, hidden_dim)

#         transformer_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=n_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=dropout_prob,
#             batch_first=True  # Important for (batch, seq, feature)
#         )
#         self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)

#         self.q_net = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),

#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),

#             nn.Linear(hidden_dim, num_actions)  # One Q-value per action
#         )

#     def forward(self, input_vec):
#         # Expect input_vec shape: (batch_size, seq_len, input_dim)
#         if input_vec.dim() != 3:
#             raise ValueError(f"Expected 3D input [batch, seq, input_dim], got {input_vec.shape}")

#         x = self.embedding(input_vec)          # -> (batch, seq, hidden_dim)
#         x = self.transformer(x)                # -> (batch, seq, hidden_dim)
#         x = x.mean(dim=1)                      # Mean pooling over sequence
#         return self.q_net(x)   

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128, dropout_prob=0.05):
        super(DQN, self).__init__()

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),


            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # nn.Linear(hidden_dim//4, hidden_dim//8),
            # nn.ReLU(),
            # nn.Dropout(dropout_prob),



            nn.Linear(hidden_dim//4, num_actions)  # One Q-value per action
        )


        # self.q_net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),

        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),

        #     nn.Linear(hidden_dim // 2, hidden_dim // 4),
        #     nn.LayerNorm(hidden_dim // 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),

        #     nn.Linear(hidden_dim // 4, hidden_dim // 8),
        #     nn.LayerNorm(hidden_dim // 8),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),

        #     nn.Linear(hidden_dim // 8, num_actions)  # One Q-value per action
        # )

    def forward(self, input_vec):
        return self.q_net(input_vec)

class DQNWithTransformer(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128, seq_len=4, n_heads=4, dropout=0.1):
        super(DQNWithTransformer, self).__init__()

        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True  # Requires PyTorch 1.10+
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # -> (batch_size, seq_len, hidden_dim)
        x = self.transformer(x)  # -> (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)        # Aggregate over sequence (mean pooling)
        return self.fc_out(x)    # -> (batch_size, num_actions)

def train_on_batch(model, optimizer, batch, gamma=0.99):
    hist_batch = torch.cat([b["state"] for b in batch])
    actions_batch = torch.cat([b["actions"] for b in batch])
    next_hist_batch = torch.cat([b["next_state"] for b in batch])
    next_actions_batch = torch.cat([b["next_actions"] for b in batch])
    
    action_indices = torch.tensor([b["action_index"] for b in batch]).unsqueeze(1)
    rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
    dones = torch.tensor([b["done"] for b in batch], dtype=torch.bool)
    batch_input = torch.cat([hist_batch, actions_batch.view(actions_batch.size(0), -1)], dim=1)  # flatten actions
    q_values = model(batch_input)
    # q_values = model(hist_batch, actions_batch)
    q_chosen = q_values.gather(1, action_indices).squeeze(1)

    with torch.no_grad():
        batch_input_next = torch.cat([next_hist_batch, next_actions_batch.view(next_actions_batch.size(0), -1)], dim=1)
        # next_q_values = model(next_hist_batch, next_actions_batch)  # (batch_size, num_actions)
        next_q_values = model(batch_input_next)  # (batch_size, num_actions)
        next_q_max = next_q_values.max(dim=1)[0]  # shape: (batch_size,)
        target_q = rewards + gamma * next_q_max * (~dones)

    loss = torch.nn.SmoothL1Loss()(q_chosen, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print("Sample rewards:", rewards[:5])
    # print("Sample dones:", dones[:5])
    # print("Sample target Q:", target_q[:5])
    # print("Sample chosen Q:", q_chosen[:5])
    # print("Loss:", loss.item())

    return loss.item()

def train_model(model, dataset, prob_dict,rules,num_episodes=1000, batch_size=32, gamma=0.99):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    buffer = ReplayBuffer(capacity=5000)

    # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    for episode in range(num_episodes):
        # Random starting sequence
        chord_seq = random.choice(dataset)
        env = Environment(chord_seq, rules, prob_dict)
        env.add_model(model)
        loss = None
        while not env.is_terminal():
            transition = env.step(epsilon=0.1)
            if transition and transition["next_actions"].numel() > 0:
                buffer.add(transition)

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = train_on_batch(model, optimizer, batch, gamma)
        # scheduler.step()
        if episode % 10 == 0:
            if loss:
                print(f"Episode {episode} complete | Last loss: {loss:.4f}")

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
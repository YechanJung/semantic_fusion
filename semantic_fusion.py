import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Physical State Encoder
class PhysicalStateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(PhysicalStateEncoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x

# Semantic Output Encoder
class SemanticOutputEncoder(nn.Module):
    def __init__(self, semantic_dim, hidden_dim):
        super(SemanticOutputEncoder, self).__init__()
        self.fc1 = nn.Linear(semantic_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, semantic):
        x = F.relu(self.fc1(semantic))
        x = F.relu(self.fc2(x))
        return x

# Cross-Modal Attention Mechanism
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.query_physical = nn.Linear(hidden_dim, hidden_dim)
        self.key_semantic = nn.Linear(hidden_dim, hidden_dim)
        self.value_physical = nn.Linear(hidden_dim, hidden_dim)
        self.value_semantic = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, physical_feat, semantic_feat):
        q = self.query_physical(physical_feat)
        k = self.key_semantic(semantic_feat)
        
        # Calculate attention weights
        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))
        attention_weights = F.softmax(attention, dim=-1)
        
        # Calculate weighted sum
        physical_value = self.value_physical(physical_feat)
        semantic_value = self.value_semantic(semantic_feat)
        
        alpha_physical = attention_weights[:, 0, 0].unsqueeze(1)
        alpha_semantic = attention_weights[:, 0, 1].unsqueeze(1)
        
        # Fused representation
        fused = alpha_physical * physical_value + alpha_semantic * semantic_value
        
        # Add residual connection
        concat = torch.cat([physical_feat, semantic_feat], dim=1)
        fused = fused + self.fc(concat)
        
        return fused, alpha_physical, alpha_semantic

# Actor-Critic Network (Policy and Value Function)
class SemanticFusionPPO(nn.Module):
    def __init__(self, state_dim, semantic_dim, action_dim, hidden_dim=128):
        super(SemanticFusionPPO, self).__init__()
        
        # Encoders
        self.physical_encoder = PhysicalStateEncoder(state_dim, hidden_dim)
        self.semantic_encoder = SemanticOutputEncoder(semantic_dim, hidden_dim)
        
        # Fusion mechanism
        self.fusion = CrossModalAttention(hidden_dim)
        
        # Policy network (Actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network (Critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state, semantic):
        # Encode each modality
        physical_feat = self.physical_encoder(state)
        semantic_feat = self.semantic_encoder(semantic)
        
        # Modality fusion
        fused_feat, alpha_physical, alpha_semantic = self.fusion(physical_feat, semantic_feat)
        
        # Calculate policy and value
        action_mean = self.policy(fused_feat)
        value = self.value(fused_feat)
        
        # Policy distribution
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        return dist, value, alpha_physical, alpha_semantic
        
# PPO Algorithm
class SemanticFusionPPOAgent:
    def __init__(self, state_dim, semantic_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.model = SemanticFusionPPO(state_dim, semantic_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
    def get_action(self, state, semantic):
        state = torch.FloatTensor(state).unsqueeze(0)
        semantic = torch.FloatTensor(semantic).unsqueeze(0)
        
        dist, _, _, _ = self.model(state, semantic)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().numpy().squeeze(), log_prob.detach().numpy()
        
    def update(self, trajectories):
        # Prepare data
        states = torch.FloatTensor([t[0] for t in trajectories])
        semantics = torch.FloatTensor([t[1] for t in trajectories])
        actions = torch.FloatTensor([t[2] for t in trajectories])
        rewards = torch.FloatTensor([t[3] for t in trajectories])
        next_states = torch.FloatTensor([t[4] for t in trajectories])
        next_semantics = torch.FloatTensor([t[5] for t in trajectories])
        dones = torch.FloatTensor([t[6] for t in trajectories])
        old_log_probs = torch.FloatTensor([t[7] for t in trajectories])
        
        # Calculate advantages
        with torch.no_grad():
            _, next_values, _, _ = self.model(next_states, next_semantics)
            next_values = next_values.squeeze()
            
            _, values, _, _ = self.model(states, semantics)
            values = values.squeeze()
            
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values
        
        # Perform PPO updates
        for _ in range(self.epochs):
            # Calculate log probabilities and entropy from current policy
            dist, values, alpha_physical, alpha_semantic = self.model(states, semantics)
            values = values.squeeze()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            
            # Calculate PPO ratio
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            
            # Calculate loss (policy loss, value loss, entropy bonus)
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, targets)
            
            # Attention regularization loss for modality balance
            attention_loss = F.mse_loss(alpha_physical + alpha_semantic, 
                                        torch.ones_like(alpha_physical + alpha_semantic))
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + 0.1 * attention_loss
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return policy_loss.item(), value_loss.item(), entropy.item()

# Example training process
def train_semantic_fusion_ppo(env, agent, max_episodes=1000, max_steps=1000):
    for episode in range(max_episodes):
        state, info = env.reset()
        
        # Separate physical state and perception model outputs
        physical_state = state[:6]  # e.g., [x, y, z, vx, vy, vz]
        semantic_output = state[6:]  # e.g., [detection_x, detection_y, confidence]
        
        episode_reward = 0
        trajectories = []
        
        for step in range(max_steps):
            # Select action
            action, log_prob = agent.get_action(physical_state, semantic_output)
            
            # Interact with environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Separate next state
            next_physical_state = next_state[:6]
            next_semantic_output = next_state[6:]
            
            # Store experience
            trajectories.append([
                physical_state, semantic_output, action, reward, 
                next_physical_state, next_semantic_output, done, log_prob
            ])
            
            # Update state
            physical_state = next_physical_state
            semantic_output = next_semantic_output
            episode_reward += reward
            
            if done:
                break
        
        # Batch update
        if trajectories:
            policy_loss, value_loss, entropy = agent.update(trajectories)
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, " 
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
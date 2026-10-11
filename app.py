"""
LogicRLCD v9.0: Complete Implementation
A Neuro-Symbolic Reinforcement Learning Framework with Contrastive Decision Transformers
and Differentiable Logic for Safe, Interpretable Decision-Making.
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Configuration and Hyperparameters
# =============================================================================

@dataclass
class Config:
    # Environment parameters
    state_dim: int = 128
    action_dim: int = 5  # up, down, left, right, hover
    max_episodes: int = 1000
    max_steps: int = 100
    
    # Model parameters
    hidden_dim: int = 256
    latent_dim: int = 64
    num_predicates: int = 20
    num_rules: int = 10
    num_heads: int = 4
    
    # Training parameters
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_logic: float = 0.3
    lambda_contrastive: float = 0.5
    batch_size: int = 64
    buffer_size: int = 10000
    
    # Safety parameters
    safety_threshold: float = 0.9
    constraint_violation_penalty: float = -10.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Logical Foundation: Propositional and Predicate Logic
# =============================================================================

class Predicate:
    """Represents a first-order logic predicate."""
    
    def __init__(self, name: str, arity: int):
        self.name = name
        self.arity = arity  # Number of arguments
        self.embedding = nn.Parameter(torch.randn(256) * 0.1)
    
    def __call__(self, *args):
        """Create a grounded predicate instance."""
        return GroundedPredicate(self, args)

class GroundedPredicate:
    """A predicate with specific arguments."""
    
    def __init__(self, predicate: Predicate, args: tuple):
        self.predicate = predicate
        self.args = args
    
    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.predicate.name}({args_str})"

class LogicalFormula:
    """Represents a logical formula (can be complex)."""
    
    def __init__(self, formula_type: str, *children):
        self.type = formula_type  # 'predicate', 'and', 'or', 'not', 'implies', 'forall', 'exists'
        self.children = children
    
    def evaluate(self, state: Dict[str, float], fuzzy: bool = True) -> float:
        """Evaluate the formula given a state."""
        if self.type == 'predicate':
            pred_name = self.children[0]
            return state.get(pred_name, 0.0)
        
        elif self.type == 'and':
            if fuzzy:
                # Product t-norm for conjunction
                result = 1.0
                for child in self.children:
                    result *= child.evaluate(state, fuzzy)
                return result
            else:
                return all(child.evaluate(state, fuzzy) for child in self.children)
        
        elif self.type == 'or':
            if fuzzy:
                # Probabilistic sum for disjunction
                result = 1.0
                for child in self.children:
                    result = result + child.evaluate(state, fuzzy) - result * child.evaluate(state, fuzzy)
                return result
            else:
                return any(child.evaluate(state, fuzzy) for child in self.children)
        
        elif self.type == 'not':
            child_val = self.children[0].evaluate(state, fuzzy)
            return 1.0 - child_val if fuzzy else not child_val
        
        elif self.type == 'implies':
            antecedent = self.children[0].evaluate(state, fuzzy)
            consequent = self.children[1].evaluate(state, fuzzy)
            if fuzzy:
                # Goguen implication: a → b = min(1, b/a)
                if antecedent == 0:
                    return 1.0
                return min(1.0, consequent / antecedent)
            else:
                return (not antecedent) or consequent
        
        elif self.type == 'forall':
            # Universal quantification
            var_name = self.children[0]
            domain = self.children[1]
            formula = self.children[2]
            
            if fuzzy:
                result = 1.0
                for item in domain:
                    new_state = state.copy()
                    new_state[var_name] = item
                    result *= formula.evaluate(new_state, fuzzy)
                return result
            else:
                return all(formula.evaluate({**state, var_name: item}, fuzzy) for item in domain)
        
        elif self.type == 'exists':
            # Existential quantification
            var_name = self.children[0]
            domain = self.children[1]
            formula = self.children[2]
            
            if fuzzy:
                result = 0.0
                for item in domain:
                    new_state = state.copy()
                    new_state[var_name] = item
                    result = max(result, formula.evaluate(new_state, fuzzy))
                return result
            else:
                return any(formula.evaluate({**state, var_name: item}, fuzzy) for item in domain)
        
        return 0.0

class KnowledgeBase:
    """Manages logical rules and constraints."""
    
    def __init__(self):
        self.predicates: Dict[str, Predicate] = {}
        self.rules: List[LogicalFormula] = []
        self.constraints: List[LogicalFormula] = []
    
    def add_predicate(self, name: str, arity: int = 0):
        """Add a predicate to the knowledge base."""
        if name not in self.predicates:
            self.predicates[name] = Predicate(name, arity)
    
    def add_rule(self, rule: LogicalFormula):
        """Add a logical rule."""
        self.rules.append(rule)
    
    def add_constraint(self, constraint: LogicalFormula):
        """Add a safety constraint."""
        self.constraints.append(constraint)
    
    def verify_state(self, state: Dict[str, float]) -> Tuple[bool, float]:
        """Verify if state satisfies all constraints."""
        total_satisfaction = 1.0
        all_satisfied = True
        
        for constraint in self.constraints:
            satisfaction = constraint.evaluate(state, fuzzy=True)
            total_satisfaction *= satisfaction
            if satisfaction < 0.5:
                all_satisfied = False
        
        return all_satisfied, total_satisfaction

# =============================================================================
# Differentiable Logic Layer
# =============================================================================

class DifferentiableLogicLayer(nn.Module):
    """Differentiable implementation of logical operations using Product t-norm."""
    
    def __init__(self, input_dim: int, num_predicates: int):
        super().__init__()
        self.predicate_embeddings = nn.Parameter(
            torch.randn(num_predicates, input_dim) * 0.1
        )
        self.logic_ops = {
            'AND': self.fuzzy_and,
            'OR': self.fuzzy_or,
            'NOT': self.fuzzy_not,
            'IMPLIES': self.fuzzy_implies,
            'FORALL': self.fuzzy_forall,
            'EXISTS': self.fuzzy_exists
        }
    
    def forward(self, x: torch.Tensor, formulas: List[LogicalFormula]) -> torch.Tensor:
        """Evaluate logical formulas differentiably."""
        results = []
        for formula in formulas:
            result = self.evaluate_formula(x, formula)
            results.append(result)
        return torch.stack(results, dim=-1)
    
    def evaluate_formula(self, x: torch.Tensor, formula: LogicalFormula) -> torch.Tensor:
        """Evaluate a logical formula on input tensor."""
        if formula.type == 'predicate':
            # Get predicate embedding
            pred_idx = hash(formula.children[0]) % self.predicate_embeddings.size(0)
            pred_emb = self.predicate_embeddings[pred_idx]
            # Compute truth value as dot product
            return torch.sigmoid(torch.sum(x * pred_emb, dim=-1))
        
        elif formula.type == 'and':
            result = torch.ones(x.size(0), device=x.device)
            for child in formula.children:
                child_val = self.evaluate_formula(x, child)
                result = result * child_val  # Product t-norm
            return result
        
        elif formula.type == 'or':
            result = torch.zeros(x.size(0), device=x.device)
            for child in formula.children:
                child_val = self.evaluate_formula(x, child)
                result = result + child_val - result * child_val  # Probabilistic sum
            return result
        
        elif formula.type == 'not':
            child_val = self.evaluate_formula(x, formula.children[0])
            return 1.0 - child_val
        
        elif formula.type == 'implies':
            antecedent = self.evaluate_formula(x, formula.children[0])
            consequent = self.evaluate_formula(x, formula.children[1])
            # Goguen implication
            return torch.where(
                antecedent > 0,
                torch.clamp(consequent / (antecedent + 1e-8), max=1.0),
                torch.ones_like(antecedent)
            )
        
        return torch.zeros(x.size(0), device=x.device)
    
    def fuzzy_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b
    
    def fuzzy_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - a * b
    
    def fuzzy_not(self, a: torch.Tensor) -> torch.Tensor:
        return 1.0 - a
    
    def fuzzy_implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(a > 0, torch.clamp(b / (a + 1e-8), max=1.0), torch.ones_like(a))
    
    def fuzzy_forall(self, values: torch.Tensor) -> torch.Tensor:
        """Differentiable universal quantification using product."""
        return torch.prod(values, dim=-1)
    
    def fuzzy_exists(self, values: torch.Tensor) -> torch.Tensor:
        """Differentiable existential quantification using max."""
        return torch.max(values, dim=-1)[0]

# =============================================================================
# Neural Components
# =============================================================================

class StateEncoder(nn.Module):
    """Encodes environment states into latent representations."""
    
    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

class ActionEncoder(nn.Module):
    """Encodes actions into latent space."""
    
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(action_dim, latent_dim)
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.embedding(action)

class WorldModel(nn.Module):
    """Predicts future states and rewards with logical verification."""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.GRU(state_dim, latent_dim, batch_first=True)
        self.dynamics = nn.GRU(latent_dim + action_dim, latent_dim, batch_first=True)
        self.decoder = nn.Linear(latent_dim, state_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
    
    def forward(self, state_seq: torch.Tensor, action_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode initial state
        _, hidden = self.encoder(state_seq)
        
        # Predict future states
        predicted_states = []
        rewards = []
        
        for t in range(action_seq.size(1)):
            # Combine latent state with action
            combined = torch.cat([hidden.squeeze(0), action_seq[:, t]], dim=-1)
            
            # Predict next latent state
            _, hidden = self.dynamics(combined.unsqueeze(1), hidden)
            
            # Decode to state space
            predicted_state = self.decoder(hidden.squeeze(0))
            predicted_states.append(predicted_state)
            
            # Predict reward
            reward = self.reward_head(hidden.squeeze(0))
            rewards.append(reward)
        
        return torch.stack(predicted_states, dim=1), torch.stack(rewards, dim=1)
    
    def imagine_future(self, state: torch.Tensor, actions: List[int]) -> torch.Tensor:
        """Imagine future states given a sequence of actions."""
        state_tensor = state.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        action_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            predicted_states, _ = self.forward(state_tensor, action_tensor)
        
        return predicted_states.squeeze(0)

# =============================================================================
# Contrastive Decision Transformer (ConDT)
# =============================================================================

class ContrastiveDecisionTransformer(nn.Module):
    """Learns discriminable embeddings for decision-making through contrastive learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.action_encoder = nn.Embedding(action_dim, hidden_dim)
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = 0.1
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute contrastive logits for state-action pairs."""
        # Encode state sequence
        state_emb, _ = self.state_encoder(states)
        
        # Encode actions
        action_emb = self.action_encoder(actions)
        
        # Compute contrastive logits
        logits = self.contrastive_head(
            torch.cat([state_emb, action_emb], dim=-1)
        )
        
        return logits
    
    def compute_contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute InfoNCE loss for contrastive learning."""
        states = batch['states']
        actions = batch['actions']
        
        # Positive pairs: (state, correct_action)
        pos_logits = self(states, actions)
        
        # Negative pairs: (state, random_action)
        neg_actions = actions[torch.randperm(len(actions))]
        neg_logits = self(states, neg_actions)
        
        # InfoNCE loss
        pos_loss = -torch.log(
            torch.exp(pos_logits / self.temperature) / 
            (torch.exp(pos_logits / self.temperature) + 
             torch.exp(neg_logits / self.temperature))
        )
        
        return pos_loss.mean()

# =============================================================================
# Differentiable Logic Policy
# =============================================================================

class DifferentiableLogicPolicy(nn.Module):
    """Policy network with differentiable logic constraints."""
    
    def __init__(self, state_dim: int, action_dim: int, num_predicates: int):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Differentiable logic layer
        self.logic_layer = DifferentiableLogicLayer(128, num_predicates)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor, formulas: Optional[List[LogicalFormula]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the policy network."""
        # Encode state
        z = self.state_encoder(state)
        
        # Evaluate logical constraints if provided
        logic_values = None
        if formulas:
            logic_values = self.logic_layer(z, formulas)
        
        # Compute action logits
        action_logits = self.policy_head(z)
        
        # Compute state value
        value = self.value_head(z)
        
        return action_logits, value, logic_values
    
    def select_action(self, state: torch.Tensor, formulas: Optional[List[LogicalFormula]] = None, 
                     safety_mask: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using the policy."""
        action_logits, value, logic_values = self.forward(state, formulas)
        
        # Apply safety mask if provided
        if safety_mask is not None:
            action_logits = action_logits + (safety_mask - 1) * 1e6
        
        # Sample action from distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value

# =============================================================================
# Safety Shield with Neuro-Symbolic Verification
# =============================================================================

class SafetyShield:
    """Verifies action safety using neuro-symbolic methods."""
    
    def __init__(self, knowledge_base: KnowledgeBase, threshold: float = 0.9):
        self.kb = knowledge_base
        self.threshold = threshold
        self.cache = {}
        self.stats = {
            'total_checks': 0,
            'cache_hits': 0,
            'violations': 0
        }
    
    def verify_action(self, state: Dict[str, float], action: int) -> Tuple[bool, float]:
        """Verify if action is safe given current state."""
        self.stats['total_checks'] += 1
        
        # Check cache
        cache_key = self._generate_cache_key(state, action)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # Simulate action effects (simplified)
        next_state = self._simulate_action(state, action)
        
        # Verify constraints
        is_safe, satisfaction = self.kb.verify_state(next_state)
        
        # Cache result
        result = (is_safe, satisfaction)
        self.cache[cache_key] = result
        
        if not is_safe:
            self.stats['violations'] += 1
        
        return result
    
    def _generate_cache_key(self, state: Dict[str, float], action: int) -> Tuple:
        """Generate cache key for verification."""
        state_tuple = tuple(sorted(state.items()))
        return (state_tuple, action)
    
    def _simulate_action(self, state: Dict[str, float], action: int) -> Dict[str, float]:
        """Simulate action effects on state."""
        next_state = state.copy()
        
        # Simplified simulation for drone navigation
        if action == 0:  # up
            next_state['altitude'] = min(1.0, state.get('altitude', 0.5) + 0.1)
        elif action == 1:  # down
            next_state['altitude'] = max(0.0, state.get('altitude', 0.5) - 0.1)
        elif action == 2:  # left
            next_state['x_pos'] = max(-1.0, state.get('x_pos', 0.0) - 0.1)
        elif action == 3:  # right
            next_state['x_pos'] = min(1.0, state.get('x_pos', 0.0) + 0.1)
        # action == 4: hover (no change)
        
        # Update logical predicates based on new state
        next_state['NearObstacle'] = 1.0 if abs(next_state.get('x_pos', 0.0)) > 0.8 else 0.0
        next_state['AltitudeSafe'] = 1.0 if 0.2 < next_state.get('altitude', 0.5) < 0.8 else 0.0
        
        return next_state
    
    def get_safety_mask(self, state: Dict[str, float], action_dim: int) -> torch.Tensor:
        """Generate safety mask for actions."""
        mask = torch.ones(action_dim)
        
        for action in range(action_dim):
            is_safe, _ = self.verify_action(state, action)
            if not is_safe:
                mask[action] = 0.0
        
        return mask

# =============================================================================
# Experience Replay Buffer
# =============================================================================

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    logic_state: Dict[str, float]
    info: Dict[str, Any] = field(default_factory=dict)

class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# LogicRLCD Trainer
# =============================================================================

class LogicRLCDTrainer:
    """Main trainer for LogicRLCD framework."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize knowledge base with logical constraints
        self.kb = self._initialize_knowledge_base()
        
        # Initialize models
        self.policy = DifferentiableLogicPolicy(
            config.state_dim, config.action_dim, config.num_predicates
        ).to(self.device)
        
        self.world_model = WorldModel(
            config.state_dim, config.action_dim, config.latent_dim
        ).to(self.device)
        
        self.contrastive_model = ContrastiveDecisionTransformer(
            config.state_dim, config.action_dim
        ).to(self.device)
        
        # Initialize safety shield
        self.safety_shield = SafetyShield(self.kb, config.safety_threshold)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) +
            list(self.world_model.parameters()) +
            list(self.contrastive_model.parameters()),
            lr=config.lr
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training statistics
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'constraint_violations': [],
            'logic_losses': [],
            'contrastive_losses': [],
            'world_model_losses': []
        }
    
    def _initialize_knowledge_base(self) -> KnowledgeBase:
        """Initialize knowledge base with logical constraints."""
        kb = KnowledgeBase()
        
        # Add predicates
        kb.add_predicate('NearObstacle', 0)
        kb.add_predicate('AltitudeSafe', 0)
        kb.add_predicate('BatteryLow', 0)
        kb.add_predicate('InNoFlyZone', 0)
        kb.add_predicate('PathClear', 0)
        
        # Add constraints (simplified for example)
        # Constraint 1: If near obstacle, must maintain safe altitude
        constraint1 = LogicalFormula(
            'implies',
            LogicalFormula('predicate', 'NearObstacle'),
            LogicalFormula('predicate', 'AltitudeSafe')
        )
        kb.add_constraint(constraint1)
        
        # Constraint 2: Never enter no-fly zones
        constraint2 = LogicalFormula(
            'not',
            LogicalFormula('predicate', 'InNoFlyZone')
        )
        kb.add_constraint(constraint2)
        
        return kb
    
    def train_episode(self, env) -> Dict[str, float]:
        """Train for one episode."""
        state = env.reset()
        logic_state = self._extract_logic_state(state)
        
        episode_reward = 0
        episode_length = 0
        constraint_violations = 0
        
        for step in range(self.config.max_steps):
            # Get safety mask
            safety_mask = self.safety_shield.get_safety_mask(logic_state, self.config.action_dim)
            
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.select_action(
                state_tensor, self.kb.constraints, safety_mask.to(self.device)
            )
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            next_logic_state = self._extract_logic_state(next_state)
            
            # Check constraint satisfaction
            is_safe, _ = self.kb.verify_state(next_logic_state)
            if not is_safe:
                constraint_violations += 1
                reward += self.config.constraint_violation_penalty
            
            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                logic_state=logic_state
            )
            self.replay_buffer.push(experience)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            # Move to next state
            state = next_state
            logic_state = next_logic_state
            
            if done:
                break
        
        # Train models
        if len(self.replay_buffer) > self.config.batch_size:
            losses = self._train_step()
            self.stats['logic_losses'].append(losses.get('logic_loss', 0))
            self.stats['contrastive_losses'].append(losses.get('contrastive_loss', 0))
            self.stats['world_model_losses'].append(losses.get('world_model_loss', 0))
        
        # Update episode statistics
        self.stats['episode_rewards'].append(episode_reward)
        self.stats['episode_lengths'].append(episode_length)
        self.stats['constraint_violations'].append(constraint_violations)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'violations': constraint_violations
        }
    
    def _train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        # Sample batch
        experiences = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Train policy
        policy_loss = self._compute_policy_loss(states, actions, rewards, next_states, dones)
        
        # Train world model
        world_model_loss = self._compute_world_model_loss(states, actions, next_states, rewards)
        
        # Train contrastive model
        contrastive_loss = self._compute_contrastive_loss(states, actions, next_states)
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.lambda_logic * world_model_loss +
            self.config.lambda_contrastive * contrastive_loss
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'logic_loss': world_model_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'world_model_loss': world_model_loss.item()
        }
    
    def _compute_policy_loss(self, states, actions, rewards, next_states, dones):
        """Compute policy gradient loss."""
        # Get current policy outputs
        action_logits, values, logic_values = self.policy(states, self.kb.constraints)
        
        # Compute advantages (simplified)
        with torch.no_grad():
            _, next_values, _ = self.policy(next_states, self.kb.constraints)
            target_values = rewards + self.config.gamma * next_values.squeeze(-1) * (1 - dones)
            advantages = target_values - values.squeeze(-1)
        
        # Policy gradient loss
        dist = Categorical(F.softmax(action_logits, dim=-1))
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value function loss
        value_loss = F.mse_loss(values.squeeze(-1), target_values.detach())
        
        # Logic consistency loss
        logic_loss = torch.tensor(0.0, device=self.device)
        if logic_values is not None:
            # Penalize logic violations
            logic_loss = (1.0 - logic_values).mean()
        
        return policy_loss + 0.5 * value_loss + 0.3 * logic_loss
    
    def _compute_world_model_loss(self, states, actions, next_states, rewards):
        """Compute world model prediction loss."""
        # Prepare sequences
        state_seq = states.unsqueeze(1)  # Add sequence dimension
        action_seq = F.one_hot(actions, self.config.action_dim).float().unsqueeze(1)
        
        # Forward pass through world model
        predicted_states, predicted_rewards = self.world_model(state_seq, action_seq)
        
        # Compute losses
        state_loss = F.mse_loss(predicted_states.squeeze(1), next_states)
        reward_loss = F.mse_loss(predicted_rewards.squeeze(1), rewards)
        
        return state_loss + reward_loss
    
    def _compute_contrastive_loss(self, states, actions, next_states):
        """Compute contrastive learning loss."""
        # Prepare batch for contrastive learning
        batch = {
            'states': states.unsqueeze(1),  # Add sequence dimension
            'actions': actions.unsqueeze(1)
        }
        
        return self.contrastive_model.compute_contrastive_loss(batch)
    
    def _extract_logic_state(self, state: np.ndarray) -> Dict[str, float]:
        """Extract logical predicates from state."""
        # This is a simplified extraction for demonstration
        # In practice, this would be domain-specific
        logic_state = {
            'NearObstacle': 1.0 if abs(state[0]) > 0.8 else 0.0,
            'AltitudeSafe': 1.0 if 0.2 < state[1] < 0.8 else 0.0,
            'BatteryLow': 1.0 if state[2] < 0.2 else 0.0,
            'InNoFlyZone': 1.0 if state[0] > 0.9 else 0.0,
            'PathClear': 1.0 if abs(state[0]) < 0.7 else 0.0
        }
        return logic_state
    
    def train(self, env, num_episodes: Optional[int] = None):
        """Train the agent."""
        num_episodes = num_episodes or self.config.max_episodes
        
        with tqdm(total=num_episodes, desc="Training") as pbar:
            for episode in range(num_episodes):
                stats = self.train_episode(env)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Reward': f"{stats['reward']:.2f}",
                    'Violations': stats['violations']
                })
                
                # Log progress
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(self.stats['episode_rewards'][-100:])
                    avg_violations = np.mean(self.stats['constraint_violations'][-100:])
                    print(f"\nEpisode {episode + 1}")
                    print(f"  Avg Reward: {avg_reward:.2f}")
                    print(f"  Avg Violations: {avg_violations:.2f}")
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent."""
        rewards = []
        violations = []
        
        for _ in range(num_episodes):
            state = env.reset()
            logic_state = self._extract_logic_state(state)
            
            episode_reward = 0
            episode_violations = 0
            
            for step in range(self.config.max_steps):
                # Get safety mask
                safety_mask = self.safety_shield.get_safety_mask(logic_state, self.config.action_dim)
                
                # Select action (greedy)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_logits, _, _ = self.policy(state_tensor, self.kb.constraints)
                action = action_logits.argmax(dim=-1).item()
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                next_logic_state = self._extract_logic_state(next_state)
                
                # Check constraints
                is_safe, _ = self.kb.verify_state(next_logic_state)
                if not is_safe:
                    episode_violations += 1
                
                episode_reward += reward
                state = next_state
                logic_state = next_logic_state
                
                if done:
                    break
            
            rewards.append(episode_reward)
            violations.append(episode_violations)
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_violations': np.mean(violations),
            'success_rate': np.mean([r > 0 for r in rewards])
        }

# =============================================================================
# Example Environment: Drone Navigation
# =============================================================================

class DroneNavigationEnv:
    """Simplified drone navigation environment with logical constraints."""
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.action_dim = 5  # up, down, left, right, hover
        self.state_dim = 6  # x, y, altitude, battery, velocity, distance_to_goal
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.position = np.array([0.0, 0.0, 0.5])  # x, y, altitude
        self.goal = np.array([0.9, 0.9, 0.5])
        self.battery = 1.0
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action."""
        self.step_count += 1
        
        # Action effects
        if action == 0:  # up (increase altitude)
            self.position[2] = min(1.0, self.position[2] + 0.1)
        elif action == 1:  # down (decrease altitude)
            self.position[2] = max(0.0, self.position[2] - 0.1)
        elif action == 2:  # left
            self.position[0] = max(-1.0, self.position[0] - 0.1)
        elif action == 3:  # right
            self.position[0] = min(1.0, self.position[0] + 0.1)
        # action == 4: hover
        
        # Battery consumption
        self.battery = max(0.0, self.battery - 0.01)
        
        # Compute reward
        distance_to_goal = np.linalg.norm(self.position - self.goal)
        reward = -distance_to_goal  # Negative distance as reward
        
        # Check if goal reached
        if distance_to_goal < 0.1:
            reward += 10.0
            done = True
        else:
            done = False
        
        # Check timeout
        if self.step_count >= 100:
            done = True
        
        # Additional info
        info = {
            'distance_to_goal': distance_to_goal,
            'battery': self.battery,
            'position': self.position.copy()
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state."""
        state = np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            self.battery,
            np.linalg.norm(self.velocity),
            np.linalg.norm(self.position - self.goal)
        ])
        return state
    
    def render(self):
        """Render environment (simplified)."""
        print(f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")
        print(f"Battery: {self.battery:.2f}")
        print(f"Distance to goal: {np.linalg.norm(self.position - self.goal):.2f}")

# =============================================================================
# Visualization and Analysis
# =============================================================================

class Visualizer:
    """Visualize training progress and results."""
    
    @staticmethod
    def plot_training_stats(stats: Dict[str, List[float]]):
        """Plot training statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0, 0].plot(stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Constraint violations
        axes[0, 1].plot(stats['constraint_violations'])
        axes[0, 1].set_title('Constraint Violations')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Violations')
        
        # Logic losses
        axes[1, 0].plot(stats['logic_losses'])
        axes[1, 0].set_title('Logic Consistency Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        
        # Contrastive losses
        axes[1, 1].plot(stats['contrastive_losses'])
        axes[1, 1].set_title('Contrastive Learning Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('logicrlcd_training_stats.png')
        plt.show()
    
    @staticmethod
    def visualize_navigation_path(env: DroneNavigationEnv, path: List[np.ndarray]):
        """Visualize drone navigation path."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot path
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=2)
        
        # Plot start and goal
        ax.scatter(*path[0], color='green', s=100, label='Start')
        ax.scatter(*env.goal, color='red', s=100, label='Goal')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Altitude')
        ax.set_title('Drone Navigation Path')
        ax.legend()
        
        plt.savefig('navigation_path.png')
        plt.show()

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize configuration
    config = Config(
        state_dim=6,
        action_dim=5,
        max_episodes=500,
        max_steps=100,
        hidden_dim=256,
        latent_dim=64,
        num_predicates=20,
        num_rules=10,
        lr=3e-4,
        gamma=0.99,
        lambda_logic=0.3,
        lambda_contrastive=0.5,
        batch_size=32,
        buffer_size=5000,
        safety_threshold=0.9,
        constraint_violation_penalty=-10.0
    )
    
    # Initialize environment
    env = DroneNavigationEnv(grid_size=10)
    
    # Initialize trainer
    trainer = LogicRLCDTrainer(config)
    
    # Print configuration
    print("=" * 60)
    print("LogicRLCD v9.0: Neuro-Symbolic Reinforcement Learning")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"State Dim: {config.state_dim}")
    print(f"Action Dim: {config.action_dim}")
    print(f"Max Episodes: {config.max_episodes}")
    print(f"Batch Size: {config.batch_size}")
    print("=" * 60)
    
    # Train agent
    print("\nStarting training...")
    trainer.train(env, num_episodes=200)
    
    # Evaluate agent
    print("\nEvaluating agent...")
    eval_results = trainer.evaluate(env, num_episodes=10)
    
    print("\nEvaluation Results:")
    print(f"  Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Average Violations: {eval_results['avg_violations']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2%}")
    
    # Visualize training statistics
    print("\nGenerating visualizations...")
    Visualizer.plot_training_stats(trainer.stats)
    
    # Run demonstration episode
    print("\nRunning demonstration episode...")
    state = env.reset()
    path = [state[:3]]
    
    for step in range(config.max_steps):
        # Get safety mask
        logic_state = trainer._extract_logic_state(state)
        safety_mask = trainer.safety_shield.get_safety_mask(logic_state, config.action_dim)
        
        # Select action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
        action_logits, _, _ = trainer.policy(state_tensor, trainer.kb.constraints)
        action = action_logits.argmax(dim=-1).item()
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        path.append(next_state[:3])
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Pos={info['position']}")
        
        state = next_state
        if done:
            break
    
    # Visualize navigation path
    Visualizer.visualize_navigation_path(env, path)
    
    print("\nTraining complete!")
    print(f"Total safety checks: {trainer.safety_shield.stats['total_checks']}")
    print(f"Cache hits: {trainer.safety_shield.stats['cache_hits']}")
    print(f"Constraint violations: {trainer.safety_shield.stats['violations']}")
    
    # Save models
    torch.save({
        'policy_state_dict': trainer.policy.state_dict(),
        'world_model_state_dict': trainer.world_model.state_dict(),
        'contrastive_model_state_dict': trainer.contrastive_model.state_dict(),
        'config': config
    }, 'logicrlcd_models.pth')
    
    print("\nModels saved to 'logicrlcd_models.pth'")

if __name__ == "__main__":
    main()
import torch
import numpy as np
from torch.distributions.normal import Normal
from policy_network import Policy_Network

class SVRGOptimizer:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr
        self.snapshot_params = [p.clone().detach() for p in self.params]
        self.snapshot_grads = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.s = 10  # Number of inner iterations

    def step(self):
        with torch.no_grad():
            for p, sp, sg in zip(self.params, self.snapshot_params, self.snapshot_grads):
                if p.grad is not None and sp.grad is not None:
                    p.grad = p.grad - sg + sp.grad
                    p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def take_snapshot(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                self.snapshot_params[i].copy_(p)
                self.snapshot_grads[i].zero_()
                if p.grad is not None:
                    self.snapshot_grads[i].copy_(p.grad)

    def update_memory(self):
        self.t += 1
        if self.t % self.s == 0:
            self.take_snapshot()

class REINFORCE_SVRG:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = SVRGOptimizer(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.update_memory()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

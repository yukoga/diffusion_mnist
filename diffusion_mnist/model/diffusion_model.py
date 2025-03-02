# Copyright 2023 yukoga
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import math


def sinusoidal_embedding(timesteps, embedding_dim):
    """
    Generates sinusoidal embeddings for timesteps.

    Args:
        timesteps (torch.Tensor): Timestep tensor of shape (batch_size,).
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape
        (batch_size, embedding_dim).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


class DiffusionModel(nn.Module):
    """
    Diffusion Model class.

    This class defines a simple feed forward network model
    for predicting noise for the diffusion model.
    It predicts the noise (epsilon) given a noisy image
    (x_t) and a timestep (t).
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize the DiffusionModel.

        Args:
            input_size (int): The size of the input image
            (e.g., 784 for flattened MNIST).
            hidden_size (int): The size of the hidden layers in the network.
        """
        super().__init__()
        self.embedding_dim = hidden_size  # Store embedding dimension
        self.linear1 = nn.Linear(input_size + hidden_size, hidden_size)
        # Input size adjusted for embedding
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, input_size)

    def forward(self, x_t, t):
        """
        Forward pass of the DiffusionModel.

        Predicts the noise (epsilon) given a noisy image
        (x_t) and a timestep (t).

        Args:
            x_t (torch.Tensor): The noisy image at timestep t.
            t (torch.Tensor): The timestep tensor.

        Returns:
            torch.Tensor: The predicted noise (epsilon).
        """
        t_emb = sinusoidal_embedding(t, self.embedding_dim)
        # Apply sinusoidal embedding to timestep
        input = torch.concat([x_t, t_emb], dim=1)
        # Concatenate noisy image and timestep embedding
        h = torch.relu(self.linear1(input))
        h = torch.relu(self.linear2(h))
        eps_noise_pred = self.linear3(h)  # Predict noise
        return eps_noise_pred


if __name__ == '__main__':
    pass

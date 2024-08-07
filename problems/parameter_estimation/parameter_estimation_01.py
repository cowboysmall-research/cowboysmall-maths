
# %% 0 - import libraries
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



# %% 1 - Generate training data
a = 0.5

t = np.linspace(0, 5, 100)
y = np.exp(a * t)



# %% 2 - Simulate observations with jitter
observed_y = y + 0.1 * np.random.randn(len(t))
print(observed_y.shape)



# %% 2 - 
t_tensor = torch.from_numpy(t).type(torch.Tensor).view(-1, 1)
y_tensor = torch.from_numpy(observed_y).type(torch.Tensor).view(-1, 1)



# %% 2 - Define a neural network to approximate 'a'
class SolutionEstimationNN(nn.Module):

    def __init__(self):
        super(SolutionEstimationNN, self).__init__()
        self.a = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, t):
        return torch.exp(torch.mul(self.a(t), t))



# %% 3 - Create an instance of the neural network
model     = SolutionEstimationNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)



# %% 4 - 
for epoch in range(1000):
    # Forward pass
    estimated_y = model(t_tensor)

    # Compute the loss
    loss = criterion(estimated_y, y_tensor)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# %% 0 - Extract the estimated 'a'
estimated_y = model(t_tensor).detach().numpy()
estimated_a = model.a(t_tensor).detach().numpy()



# %% 0 - 
plt.plot(t, observed_y, label = 'Observed y')
plt.plot(t, estimated_y, label = 'Estimated y')
plt.plot(t, y, label = 'Actual y')

plt.plot(t, estimated_a, label = 'Estimated a')
plt.axhline(a, label = 'Actual a')

plt.xlabel('')
plt.ylabel('a')

plt.legend()
plt.show()


# %% 0 - import libraries
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



# %% 0 - 
plt.style.use("ggplot")



# %% 0 - 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# %% 1 - Generate training data
a = 0.5

t = np.linspace(0, 10, 200)
y = np.exp(a * t)



# %% 2 - Simulate observations with jitter
y_observed = y + 0.5 * np.random.randn(len(t))



# %% 2 - 
T = torch.from_numpy(t).type(torch.Tensor).view(-1, 1)
Y = torch.from_numpy(y_observed).type(torch.Tensor).view(-1, 1)

if torch.cuda.is_available():
    T = T.cuda()
    Y = Y.cuda()



# %% 2 - Define a neural network to approximate 'a'
class FirstOrderODE(nn.Module):

    def __init__(self):
        super(FirstOrderODE, self).__init__()
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
model     = FirstOrderODE()
model     = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)



# %% 4 - 
epochs = 1000
for epoch in range(epochs):
    out  = model(T)
    loss = criterion(out, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 9:
        print(f"Epoch {epoch + 1:>5} - MSE: {loss.item()}")



# %% 0 - Extract the estimated 'a'
y_estimate = model(T).detach().numpy()
a_estimate = model.a(T).detach().numpy()



# %% 0 - 
fig, axes = plt.subplots(1, 2, figsize = (16, 6))
fig.suptitle("ODE Estimation")

axes[0].set_title("Function")
axes[0].plot(t, y_observed, label = 'Observed y')
axes[0].plot(t, y_estimate, label = 'Estimated y')
axes[0].plot(t, y, label = 'Actual y')
axes[0].legend(loc = 'lower right')
axes[0].set_xlabel("t")
axes[0].set_ylabel("y")

axes[1].set_title("Parameter")
axes[1].axhline(a, label = 'Actual a')
axes[1].plot(t, a_estimate, label = 'Estimated a')
axes[1].legend(loc = 'lower right')
axes[1].set_xlabel("t")
axes[1].set_ylabel("a")

# %%

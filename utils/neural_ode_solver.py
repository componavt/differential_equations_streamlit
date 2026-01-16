# Conditional import for neural ODE solver to handle cases where torch is not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import numpy as np


if TORCH_AVAILABLE:
    class NeuralODE(nn.Module):
        """
        Neural Network surrogate for solving ODEs
        Takes time t as input and outputs (x(t), y(t))
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 2),  # Output x, y
            )

        def forward(self, t):
            return self.net(t)


    def train_neural_ode(t_train, x_train, y_train, epochs=2000, lr=1e-3):
        """
        Train a neural network to approximate the solution on the training interval
        
        Args:
            t_train: array of time points for training
            x_train: array of x values for training
            y_train: array of y values for training
            epochs: number of training epochs
            lr: learning rate
        
        Returns:
            Trained model
        """
        # Prepare training data
        t_train_tensor = torch.tensor(t_train[:, None], dtype=torch.float32)
        xy_train_tensor = torch.tensor(np.column_stack([x_train, y_train]), dtype=torch.float32)

        # Initialize model
        model = NeuralODE()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(t_train_tensor)
            loss = loss_fn(pred, xy_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        return model


    def predict_with_neural_ode(model, t_eval):
        """
        Predict solution using the trained neural network
        
        Args:
            model: trained NeuralODE model
            t_eval: array of time points to evaluate
        
        Returns:
            x_pred, y_pred arrays
        """
        if model is None:
            return None, None
            
        with torch.no_grad():
            t_tensor = torch.tensor(t_eval[:, None], dtype=torch.float32)
            pred = model(t_tensor).numpy()
            x_pred = pred[:, 0]
            y_pred = pred[:, 1]
            
        return x_pred, y_pred
else:
    def train_neural_ode(t_train, x_train, y_train, epochs=2000, lr=1e-3):
        """
        Placeholder function when torch is not available
        """
        print("PyTorch not available, skipping neural network training")
        return None


    def predict_with_neural_ode(model, t_eval):
        """
        Placeholder function when torch is not available
        """
        return None, None
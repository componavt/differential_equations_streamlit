import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PhysicsInformedNeuralNetwork(nn.Module):
        """
        Physics-Informed Neural Network (PINN) for learning the vector field of an ODE system.
        Instead of learning the solution t -> (x(t), y(t)), this learns the vector field (x, y) -> (dx/dt, dy/dt).
        """
        def __init__(self, hidden_size=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_size),  # Input: (x, y)
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 2),  # Output: (dx/dt, dy/dt)
            )

        def forward(self, xy):
            """
            Forward pass: (x, y) -> (dx/dt, dy/dt)
            """
            return self.net(xy)


    def train_pinn(rhs_func, x0, y0, t_train, initial_conditions=None, epochs=2000, lr=1e-3):
        """
        Train a PINN to learn the vector field of the ODE system.
        
        Args:
            rhs_func: Right-hand side function of the ODE system that returns [dx/dt, dy/dt]
            x0, y0: Initial conditions
            t_train: Time points for training
            initial_conditions: Additional initial conditions for training (optional)
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Trained PINN model
        """
        # Generate training data by evaluating the known RHS function
        # This simulates having access to the derivative values for training
        xy_train = []
        dydt_train = []
        
        # For each initial condition and time point, generate training pairs
        for t in t_train:
            # For a PINN, we want to learn the general vector field function
            # So we'll generate training points by evaluating the RHS at various (x,y) positions
            # For simplicity, we'll use the actual solution points plus some variations
            
            # Get the solution at time t using the original solver (this is for generating training data)
            # In a real PINN, we'd rely more on the physics constraints rather than exact solution points
            from scipy.integrate import solve_ivp
            sol = solve_ivp(rhs_func, (0, t), [x0, y0], method='DOP853', t_eval=[t])
            
            if sol.success and len(sol.y[0]) > 0:
                x_t, y_t = sol.y[0][-1], sol.y[1][-1]
                
                # Evaluate the RHS at this point to get the true derivatives
                true_derivatives = rhs_func(None, [x_t, y_t])
                
                # Add this as a training sample
                xy_train.append([x_t, y_t])
                dydt_train.append(true_derivatives)
        
        # Convert to tensors
        xy_tensor = torch.tensor(xy_train, dtype=torch.float32)
        dydt_tensor = torch.tensor(dydt_train, dtype=torch.float32)
        
        # Initialize model
        model = PhysicsInformedNeuralNetwork()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Predict derivatives
            pred_dydt = model(xy_tensor)
            
            # Compute loss
            loss = loss_fn(pred_dydt, dydt_tensor)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        return model


    def predict_with_pinn(model, initial_condition, t_eval):
        """
        Solve the ODE using the trained PINN by integrating the learned vector field.
        
        Args:
            model: Trained PINN model
            initial_condition: Starting point [x0, y0]
            t_eval: Time points to evaluate
            
        Returns:
            x_pred, y_pred arrays
        """
        if model is None:
            return None, None
        
        # Use scipy integrator with the learned vector field
        def learned_rhs(t, state):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
                deriv_tensor = model(state_tensor)
                derivatives = deriv_tensor.numpy().flatten()
            return derivatives
        
        from scipy.integrate import solve_ivp
        sol = solve_ivp(learned_rhs, (t_eval[0], t_eval[-1]), initial_condition, 
                       method='DOP853', t_eval=t_eval)
        
        if sol.success:
            return sol.y[0], sol.y[1]
        else:
            return None, None
else:
    def train_pinn(rhs_func, x0, y0, t_train, initial_conditions=None, epochs=2000, lr=1e-3):
        """
        Placeholder function when torch is not available
        """
        print("PyTorch not available, skipping PINN training")
        return None


    def predict_with_pinn(model, initial_condition, t_eval):
        """
        Placeholder function when torch is not available
        """
        return None, None
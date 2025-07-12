import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import math
from verification_suite import MathematicalVerification

class DendriticBranch(nn.Module):
    """Individual dendritic branch with learnable weights"""
    def __init__(self, input_size: int, branch_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, branch_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class DANNNeuron(nn.Module):
    """dANN neuron with multiple dendritic branches and gating capability"""
    def __init__(self, input_size: int, n_branches: int, branch_size: int,
                 output_size: int = 1, use_gating: bool = True):
        super().__init__()
        self.n_branches = n_branches
        self.branch_size = branch_size
        self.use_gating = use_gating

        # Create dendritic branches
        self.branches = nn.ModuleList([
            DendriticBranch(input_size, branch_size)
            for _ in range(n_branches)
        ])

        # Soma combines branch outputs
        self.soma = nn.Linear(n_branches * branch_size, output_size)

        # Initialize gating values (will be overridden by gating network)
        if use_gating:
            self.register_buffer('gate_values', torch.ones(n_branches))

    def forward(self, x: torch.Tensor, gate_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Process through each dendritic branch
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_out = branch(x)

            # Apply gating if enabled
            if self.use_gating and gate_values is not None:
                if gate_values.dim() == 1:
                    gate_val = gate_values[i]
                else:
                    gate_val = gate_values[:, i:i+1]
                branch_out = branch_out * gate_val

            branch_outputs.append(branch_out)

        # Concatenate branch outputs and pass through soma
        combined = torch.cat(branch_outputs, dim=-1)
        return self.soma(combined)

class DANNLayer(nn.Module):
    """Layer of dANN neurons"""
    def __init__(self, input_size: int, n_neurons: int, n_branches: int,
                 branch_size: int, use_gating: bool = True):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_branches = n_branches
        self.use_gating = use_gating

        self.neurons = nn.ModuleList([
            DANNNeuron(input_size, n_branches, branch_size, 1, use_gating)
            for _ in range(n_neurons)
        ])

    def forward(self, x: torch.Tensor, gate_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = []
        for i, neuron in enumerate(self.neurons):
            if gate_values is not None:
                neuron_gates = gate_values[:, i * self.n_branches:(i + 1) * self.n_branches]
            else:
                neuron_gates = None
            outputs.append(neuron(x, neuron_gates))

        return torch.cat(outputs, dim=-1)

class Hypernetwork(nn.Module):
    """Hypernetwork that generates weights for the gating network"""
    def __init__(self, context_size: int, target_param_count: int,
                 hidden_sizes: List[int] = [32, 64]):
        super().__init__()
        self.context_size = context_size
        self.target_param_count = target_param_count

        # Build the hypernetwork layers
        layers = []
        prev_size = context_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        # Final layer outputs the target parameters
        layers.append(nn.Linear(prev_size, target_param_count))

        self.network = nn.Sequential(*layers)

        # Initialize with moderate weights for context sensitivity
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

        # Initialize the final layer to be sensitive to context differences
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.xavier_normal_(final_layer.weight, gain=2.0)
            nn.init.normal_(final_layer.bias, mean=0.0, std=1.0)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)

class GatingNetwork(nn.Module):
    """Differentiable gating network that directly uses hypernetwork outputs"""
    def __init__(self, input_size: int, n_neurons: int, n_branches: int,
                 branch_size: int, hidden_size: int = 8):
        super().__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.n_branches = n_branches
        self.branch_size = branch_size
        self.hidden_size = hidden_size

        # Total number of gates needed (one per branch per neuron)
        self.n_gates = n_neurons * n_branches

        # Calculate parameter sizes for hypernetwork
        self.weight_size = input_size * self.n_gates
        self.bias_size = self.n_gates
        self.total_params = self.weight_size + self.bias_size

        # Store hypernetwork outputs (maintains gradients)
        self.hyper_weights = None
        self.hyper_bias = None

    def set_hypernetwork_outputs(self, weight_vector: torch.Tensor):
        """Store hypernetwork outputs while preserving gradients"""
        # Split weight vector into weight matrix and bias
        self.hyper_weights = weight_vector[:self.weight_size].view(self.n_gates, self.input_size)
        self.hyper_bias = weight_vector[self.weight_size:self.weight_size + self.bias_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use hypernetwork outputs directly (preserves gradients!)
        if self.hyper_weights is None or self.hyper_bias is None:
            # Fallback to zeros if not configured
            logits = torch.zeros(x.size(0), self.n_gates, device=x.device)
        else:
            logits = F.linear(x, self.hyper_weights, self.hyper_bias)

        gates = torch.sigmoid(logits)
        return gates

class HypernetworkGatedDANN(nn.Module):
    """Complete Hypernetwork-Gated dANN Architecture for Mathematical Reasoning"""
    def __init__(self, input_size: int, context_size: int,
                 main_layers: List[Tuple[int, int, int]],
                 output_size: int,
                 hypernetwork_hidden: List[int] = [32, 64],
                 gating_hidden: int = 8):
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.output_size = output_size

        # Build main network (processing neurons)
        self.main_network = nn.ModuleList()
        current_size = input_size

        for n_neurons, n_branches, branch_size in main_layers:
            layer = DANNLayer(current_size, n_neurons, n_branches, branch_size, use_gating=True)
            self.main_network.append(layer)
            current_size = n_neurons

        # Final output layer (no activation for regression)
        self.output_layer = nn.Linear(current_size, output_size)

        # Build gating network
        self.gating_networks = nn.ModuleList()
        for n_neurons, n_branches, branch_size in main_layers:
            gating_net = GatingNetwork(input_size, n_neurons, n_branches,
                                     branch_size, gating_hidden)
            self.gating_networks.append(gating_net)

        # Build hypernetwork
        total_gating_params = sum(gn.total_params for gn in self.gating_networks)
        self.hypernetwork = Hypernetwork(context_size, total_gating_params, hypernetwork_hidden)

        # Store layer configurations
        self.main_layers = main_layers

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Phase 2: Process input X through the configured network"""
        batch_size = x.size(0)

        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0).expand(batch_size, -1)

            # Check if all contexts are the same for efficiency
            contexts_same = torch.all(context == context[0])

            if contexts_same:
                # All contexts identical - process batch together
                # Generate gating weights from hypernetwork
                all_gating_weights = self.hypernetwork(context[0:1])

                # Configure all gating networks
                idx = 0
                for gating_net in self.gating_networks:
                    param_count = gating_net.total_params
                    gating_weights = all_gating_weights[0, idx:idx + param_count]
                    gating_net.set_hypernetwork_outputs(gating_weights)
                    idx += param_count

                # Forward pass through main network
                current_x = x
                for main_layer, gating_net in zip(self.main_network, self.gating_networks):
                    gate_values = gating_net(x)
                    current_x = main_layer(current_x, gate_values)

                return self.output_layer(current_x)
            else:
                # Different contexts - process individually
                outputs = []
                for i in range(batch_size):
                    sample_x = x[i:i+1]
                    sample_context = context[i:i+1]

                    # Generate gating weights for this sample's context
                    all_gating_weights = self.hypernetwork(sample_context)

                    # Configure gating networks
                    idx = 0
                    for gating_net in self.gating_networks:
                        param_count = gating_net.total_params
                        gating_weights = all_gating_weights[0, idx:idx + param_count]
                        gating_net.set_hypernetwork_outputs(gating_weights)
                        idx += param_count

                    # Forward pass
                    current_x = sample_x
                    for main_layer, gating_net in zip(self.main_network, self.gating_networks):
                        gate_values = gating_net(sample_x)
                        current_x = main_layer(current_x, gate_values)

                    sample_output = self.output_layer(current_x)
                    outputs.append(sample_output)

                return torch.cat(outputs, dim=0)
        else:
            # No context - process without gating
            current_x = x
            for main_layer in self.main_network:
                current_x = main_layer(current_x, None)
            return self.output_layer(current_x)

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the architecture"""
        total_main_params = sum(p.numel() for p in self.main_network.parameters())
        total_main_params += sum(p.numel() for p in self.output_layer.parameters())

        total_gating_params = sum(gn.total_params for gn in self.gating_networks)
        total_hyper_params = sum(p.numel() for p in self.hypernetwork.parameters())

        return {
            'main_network_params': total_main_params,
            'gating_network_params': total_gating_params,
            'hypernetwork_params': total_hyper_params,
            'total_params': total_main_params + total_hyper_params,
            'layer_configs': self.main_layers,
            'context_size': self.context_size,
            'input_size': self.input_size,
            'output_size': self.output_size
        }

def create_mathematical_functions_dataset(n_samples_per_function: int = 200, x_range: Tuple[float, float] = (-2.0, 2.0)):
    """Create multi-function mathematical reasoning dataset"""

    # Generate X values
    x_min, x_max = x_range
    base_x = torch.linspace(x_min, x_max, n_samples_per_function)

    # Function definitions
    def polynomial_fn(x):
        return x ** 2  # Quadratic

    def trigonometric_fn(x):
        return torch.sin(2 * math.pi * x / 4)  # Sine with period 4

    def step_fn(x):
        return (x > 0).float()  # Step function at x=0

    def exponential_fn(x):
        return torch.exp(x / 2)  # Scaled exponential

    # Function contexts (one-hot encoding)
    polynomial_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    trigonometric_context = torch.tensor([0.0, 1.0, 0.0, 0.0])
    step_context = torch.tensor([0.0, 0.0, 1.0, 0.0])
    exponential_context = torch.tensor([0.0, 0.0, 0.0, 1.0])

    # Generate datasets
    inputs = []
    outputs = []
    contexts = []
    function_types = []

    # Add some noise to X values for robustness
    noise_scale = (x_max - x_min) * 0.01  # 1% of range

    for func_name, func, context in [
        ("polynomial", polynomial_fn, polynomial_context),
        ("trigonometric", trigonometric_fn, trigonometric_context),
        ("step", step_fn, step_context),
        ("exponential", exponential_fn, exponential_context)
    ]:
        # Add slight noise to x values
        noisy_x = base_x + torch.randn_like(base_x) * noise_scale
        x_inputs = noisy_x.unsqueeze(1)  # Shape: [n_samples, 1]

        y_outputs = func(noisy_x).unsqueeze(1)  # Shape: [n_samples, 1]
        func_contexts = context.unsqueeze(0).expand(n_samples_per_function, -1)
        func_labels = [func_name] * n_samples_per_function

        inputs.append(x_inputs)
        outputs.append(y_outputs)
        contexts.append(func_contexts)
        function_types.extend(func_labels)

    return (torch.cat(inputs, dim=0),
            torch.cat(outputs, dim=0),
            torch.cat(contexts, dim=0),
            function_types)

def test_mathematical_reasoning():
    """Test multi-function mathematical reasoning"""
    print("="*70)
    print("MULTI-FUNCTION MATHEMATICAL REASONING CHALLENGE")
    print("="*70)

    # Create model
    model = HypernetworkGatedDANN(
        input_size=1,           # Single X value
        context_size=4,         # 4 function types
        main_layers=[(6, 3, 4)], # Single layer architecture
        output_size=1,          # Single Y value
        hypernetwork_hidden=[16, 32],
        gating_hidden=8
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dataset
    X, Y, contexts, function_types = create_mathematical_functions_dataset(200)
    print(f"Dataset created: {X.shape[0]} samples across 4 mathematical functions")

    # Define test contexts
    polynomial_context = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    trigonometric_context = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    step_context = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    exponential_context = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    # Test points for evaluation
    test_x = torch.linspace(-2, 2, 50).unsqueeze(1)

    def evaluate_functions():
        """Evaluate model on all functions"""
        model.eval()
        with torch.no_grad():
            poly_pred = model(test_x, polynomial_context.expand(50, -1))
            trig_pred = model(test_x, trigonometric_context.expand(50, -1))
            step_pred = model(test_x, step_context.expand(50, -1))
            exp_pred = model(test_x, exponential_context.expand(50, -1))

            # True functions
            x_vals = test_x.squeeze()
            poly_true = x_vals ** 2
            trig_true = torch.sin(2 * math.pi * x_vals / 4)
            step_true = (x_vals > 0).float()
            exp_true = torch.exp(x_vals / 2)

            # Calculate MSE for each function
            poly_mse = F.mse_loss(poly_pred.squeeze(), poly_true)
            trig_mse = F.mse_loss(trig_pred.squeeze(), trig_true)
            step_mse = F.mse_loss(step_pred.squeeze(), step_true)
            exp_mse = F.mse_loss(exp_pred.squeeze(), exp_true)

            return {
                'polynomial': (poly_pred.squeeze(), poly_true, poly_mse),
                'trigonometric': (trig_pred.squeeze(), trig_true, trig_mse),
                'step': (step_pred.squeeze(), step_true, step_mse),
                'exponential': (exp_pred.squeeze(), exp_true, exp_mse)
            }

    print("\nINITIAL PERFORMANCE (before training):")
    initial_results = evaluate_functions()
    for func_name, (pred, true, mse) in initial_results.items():
        print(f"{func_name.capitalize():>13}: MSE = {mse.item():.4f}")

    # Check initial gating differences
    print("\nINITIAL GATING ANALYSIS:")
    sample_x = torch.tensor([[0.0]])  # Test at x=0

    # Configure each context and get gate values
    all_gating_weights_poly = model.hypernetwork(polynomial_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_poly[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break  # Just configure first layer for analysis
    poly_gates = model.gating_networks[0](sample_x)

    all_gating_weights_trig = model.hypernetwork(trigonometric_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_trig[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    trig_gates = model.gating_networks[0](sample_x)

    all_gating_weights_step = model.hypernetwork(step_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_step[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    step_gates = model.gating_networks[0](sample_x)

    all_gating_weights_exp = model.hypernetwork(exponential_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_exp[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    exp_gates = model.gating_networks[0](sample_x)

    print(f"Polynomial gates:     {poly_gates.flatten()[:8]}")
    print(f"Trigonometric gates:  {trig_gates.flatten()[:8]}")
    print(f"Step gates:           {step_gates.flatten()[:8]}")
    print(f"Exponential gates:    {exp_gates.flatten()[:8]}")

    gate_diff = torch.stack([poly_gates, trig_gates, step_gates, exp_gates])
    pairwise_diff = torch.cdist(gate_diff.view(4, -1), gate_diff.view(4, -1)).mean()
    print(f"Average pairwise gate difference: {pairwise_diff.item():.4f}")

    # Training setup with different learning rates for different components
    main_params = list(model.main_network.parameters()) + list(model.output_layer.parameters())
    hyper_params = list(model.hypernetwork.parameters())

    # Use different learning rates - hypernetwork needs higher LR to overcome gradient dilution
    optimizer = torch.optim.Adam([
        {'params': main_params, 'lr': 0.003},
        {'params': hyper_params, 'lr': 0.01}  # Higher LR for hypernetwork
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8)

    criterion = nn.MSELoss()
    batch_size = 32
    epochs = 50

    print(f"\nStarting training for {epochs} epochs...")
    print("Using differential learning rates: Main=0.003, Hypernetwork=0.01")

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        # Shuffle data
        indices = torch.randperm(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        contexts_shuffled = contexts[indices]

        # Mini-batch training
        for i in range(0, X.shape[0], batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_Y = Y_shuffled[i:i+batch_size]
            batch_contexts = contexts_shuffled[i:i+batch_size]

            optimizer.zero_grad()

            outputs = model(batch_X, batch_contexts)
            loss = criterion(outputs, batch_Y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # Print progress with enhanced monitoring
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / (X.shape[0] // batch_size)

            # Check hypernetwork differentiation and gradient flow
            model.eval()
            with torch.no_grad():
                hyper_poly = model.hypernetwork(polynomial_context)
                hyper_trig = model.hypernetwork(trigonometric_context)
                hyper_step = model.hypernetwork(step_context)
                hyper_exp = model.hypernetwork(exponential_context)

                hyper_outputs = torch.stack([hyper_poly, hyper_trig, hyper_step, hyper_exp])
                hyper_diff = torch.cdist(hyper_outputs.view(4, -1), hyper_outputs.view(4, -1)).mean()

            # Check if hypernetwork has gradients (gradient flow check)
            hyper_grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0
                                for p in model.hypernetwork.parameters())

            model.train()
            scheduler.step(avg_loss)  # Update learning rate

            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Hyper diff = {hyper_diff:.4f}, Hyper grad = {hyper_grad_norm:.4f}")

    print("\nFINAL PERFORMANCE (after training):")
    final_results = evaluate_functions()
    for func_name, (pred, true, mse) in final_results.items():
        print(f"{func_name.capitalize():>13}: MSE = {mse.item():.6f}")

    # Demonstrate context switching
    print("\nCONTEXT SWITCHING DEMONSTRATION:")
    test_point = torch.tensor([[1.0]])  # Test at x=1.0

    poly_out = model(test_point, polynomial_context).item()
    trig_out = model(test_point, trigonometric_context).item()
    step_out = model(test_point, step_context).item()
    exp_out = model(test_point, exponential_context).item()

    print(f"Input x=1.0:")
    print(f"  Polynomial context:    y = {poly_out:.4f} (true: 1.0000)")
    print(f"  Trigonometric context: y = {trig_out:.4f} (true: {math.sin(2*math.pi/4):.4f})")
    print(f"  Step context:          y = {step_out:.4f} (true: 1.0000)")
    print(f"  Exponential context:   y = {exp_out:.4f} (true: {math.exp(0.5):.4f})")

    # Final gating analysis
    print("\nFINAL GATING ANALYSIS:")
    # Configure each context and get final gate values
    all_gating_weights_poly = model.hypernetwork(polynomial_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_poly[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    poly_gates_final = model.gating_networks[0](sample_x)

    all_gating_weights_trig = model.hypernetwork(trigonometric_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_trig[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    trig_gates_final = model.gating_networks[0](sample_x)

    all_gating_weights_step = model.hypernetwork(step_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_step[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    step_gates_final = model.gating_networks[0](sample_x)

    all_gating_weights_exp = model.hypernetwork(exponential_context)
    idx = 0
    for gating_net in model.gating_networks:
        param_count = gating_net.total_params
        gating_weights = all_gating_weights_exp[0, idx:idx + param_count]
        gating_net.set_hypernetwork_outputs(gating_weights)
        break
    exp_gates_final = model.gating_networks[0](sample_x)

    print(f"Polynomial gates:     {poly_gates_final.flatten()[:8]}")
    print(f"Trigonometric gates:  {trig_gates_final.flatten()[:8]}")
    print(f"Step gates:           {step_gates_final.flatten()[:8]}")
    print(f"Exponential gates:    {exp_gates_final.flatten()[:8]}")

    gate_diff_final = torch.stack([poly_gates_final, trig_gates_final, step_gates_final, exp_gates_final])
    pairwise_diff_final = torch.cdist(gate_diff_final.view(4, -1), gate_diff_final.view(4, -1)).mean()
    print(f"Final average pairwise gate difference: {pairwise_diff_final.item():.4f}")

    return model, final_results

# Run the mathematical reasoning test
if __name__ == "__main__":
    trained_model, results = test_mathematical_reasoning()
    verifier = MathematicalVerification(trained_model)
    verification_results = verifier.generate_verification_report(save_plots=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math

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
                 hidden_sizes: List[int] = [128, 256]):
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
        
        # Initialize with small weights for stability
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)  # Normal gain
                nn.init.constant_(module.bias, 0)
        
        # Initialize the final layer to be sensitive to context differences
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.xavier_normal_(final_layer.weight, gain=2.0)  # Larger gain for sensitivity
            nn.init.normal_(final_layer.bias, mean=0.0, std=1.0)  # Larger variance in bias
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)

class GatingNetwork(nn.Module):
    """Simplified gating network with direct weight mapping"""
    def __init__(self, input_size: int, n_neurons: int, n_branches: int, 
                 branch_size: int, hidden_size: int = 64):
        super().__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.n_branches = n_branches
        self.branch_size = branch_size
        self.hidden_size = hidden_size
        
        # Total number of gates needed (one per branch per neuron)
        self.n_gates = n_neurons * n_branches
        
        # Direct mapping: input -> gates (no hidden layer for simplicity)
        self.weight_size = input_size * self.n_gates
        self.bias_size = self.n_gates
        
        # Total parameters needed
        self.total_params = self.weight_size + self.bias_size
        
        # Placeholders for weights (will be set by hypernetwork)
        self.register_buffer('weight', torch.zeros(self.n_gates, input_size))
        self.register_buffer('bias', torch.zeros(self.n_gates))
        
    def set_weights(self, weight_vector: torch.Tensor):
        """Set weights from hypernetwork output"""
        # Weight matrix
        self.weight.data = weight_vector[:self.weight_size].view(self.n_gates, self.input_size)
        
        # Bias vector  
        self.bias.data = weight_vector[self.weight_size:self.weight_size + self.bias_size]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct linear transformation followed by sigmoid
        logits = F.linear(x, self.weight, self.bias)
        gates = torch.sigmoid(logits)
        return gates

class HypernetworkGatedDANN(nn.Module):
    """Complete Hypernetwork-Gated dANN Architecture"""
    def __init__(self, input_size: int, context_size: int, 
                 main_layers: List[Tuple[int, int, int]], 
                 output_size: int, 
                 hypernetwork_hidden: List[int] = [128, 256],
                 gating_hidden: int = 64):
        """
        Args:
            input_size: Size of input data X
            context_size: Size of context vector
            main_layers: List of (n_neurons, n_branches, branch_size) for each layer
            output_size: Final output size
            hypernetwork_hidden: Hidden layer sizes for hypernetwork
            gating_hidden: Hidden layer size for gating network
        """
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
            
        # Final output layer
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
        
    def configure_context(self, context: torch.Tensor):
        """Phase 1: Configure gating networks based on context"""
        # Generate all gating weights from hypernetwork
        all_gating_weights = self.hypernetwork(context)
        
        # Handle batch dimension - take first sample for now
        # In practice, you might want different behavior here
        if all_gating_weights.dim() > 1:
            all_gating_weights = all_gating_weights[0]  # Take first sample's weights
        
        # Distribute weights to each gating network
        idx = 0
        for i, gating_net in enumerate(self.gating_networks):
            param_count = gating_net.total_params
            gating_weights = all_gating_weights[idx:idx + param_count]
            gating_net.set_weights(gating_weights)
            idx += param_count
            
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Phase 2: Process input X through the configured network"""
        batch_size = x.size(0)
        
        # Process each sample in the batch with its own context
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0).expand(batch_size, -1)
            
            # For efficiency, check if all contexts are the same
            contexts_same = torch.all(context == context[0])
            
            if contexts_same:
                # All contexts are identical - process batch together
                self.configure_context(context[0:1])
                
                current_x = x
                for main_layer, gating_net in zip(self.main_network, self.gating_networks):
                    gate_values = gating_net(x)
                    current_x = main_layer(current_x, gate_values)
                    
                return self.output_layer(current_x)
            else:
                # Different contexts - process each sample individually
                outputs = []
                for i in range(batch_size):
                    sample_x = x[i:i+1]  # Keep batch dimension
                    sample_context = context[i:i+1]  # Keep batch dimension
                    
                    # Configure gating for this sample's context
                    self.configure_context(sample_context)
                    
                    # Process through main network with gating
                    current_x = sample_x
                    for main_layer, gating_net in zip(self.main_network, self.gating_networks):
                        # Generate gate values for this layer
                        gate_values = gating_net(sample_x)  # Use original input X
                        
                        # Process through main layer with gating
                        current_x = main_layer(current_x, gate_values)
                        
                    # Final output
                    sample_output = self.output_layer(current_x)
                    outputs.append(sample_output)
                
                return torch.cat(outputs, dim=0)
        
        else:
            # No context provided - process normally without gating
            current_x = x
            for main_layer in self.main_network:
                current_x = main_layer(current_x, None)  # No gating
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

# Example usage and testing
if __name__ == "__main__":
    # Define architecture
    input_size = 10
    context_size = 3
    main_layers = [(8, 4, 6), (4, 3, 8)]  # (n_neurons, n_branches, branch_size)
    output_size = 2
    
    # Create model
    model = HypernetworkGatedDANN(
        input_size=input_size,
        context_size=context_size,
        main_layers=main_layers,
        output_size=output_size
    )
    
    # Print architecture info
    info = model.get_architecture_info()
    print("Architecture Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 5
    x = torch.randn(batch_size, input_size)
    context = torch.randn(batch_size, context_size)
    
    # Forward pass
    output = model(x, context)
    print(f"\nInput shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {output.shape}")
    
def create_multitask_dataset(n_samples: int = 1000):
    """Create a multi-task dataset with XOR and AND tasks"""
    # Generate all possible 2-bit inputs
    base_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    
    # XOR truth table
    xor_outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    # AND truth table  
    and_outputs = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    
    # Create MORE DISTINCT contexts
    xor_context = torch.tensor([1.0, -1.0], dtype=torch.float32)  # [1, -1] = XOR task
    and_context = torch.tensor([-1.0, 1.0], dtype=torch.float32)  # [-1, 1] = AND task
    
    # Generate training data by repeating the patterns
    inputs = []
    outputs = []
    contexts = []
    
    for _ in range(n_samples // 8):  # 8 = 4 patterns × 2 tasks
        # XOR task samples
        for i in range(4):
            inputs.append(base_inputs[i])
            outputs.append(xor_outputs[i])
            contexts.append(xor_context)
            
        # AND task samples
        for i in range(4):
            inputs.append(base_inputs[i])
            outputs.append(and_outputs[i])
            contexts.append(and_context)
    
    return torch.stack(inputs), torch.stack(outputs), torch.stack(contexts)

def test_training_multitask():
    """Test training on XOR vs AND tasks"""
    print("="*60)
    print("MULTI-TASK TRAINING: XOR vs AND")
    print("="*60)
    
    # Create model optimized for this task
    model = HypernetworkGatedDANN(
        input_size=2,           # 2-bit inputs
        context_size=2,         # 2-dimensional context [XOR, AND]
        main_layers=[(4, 2, 3)],  # Single layer, simpler
        output_size=1,          # Single output
        hypernetwork_hidden=[16, 32],  # Much smaller hypernetwork
        gating_hidden=8         # Much smaller gating network
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataset
    X, y, contexts = create_multitask_dataset(800)  # Smaller dataset
    print(f"Dataset created: {X.shape[0]} samples")
    
    # Test initial performance
    xor_context = torch.tensor([[1.0, -1.0]])   # More distinct contexts
    and_context = torch.tensor([[-1.0, 1.0]])   # More distinct contexts
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    
    def test_performance():
        model.eval()
        with torch.no_grad():
            xor_pred = torch.sigmoid(model(test_inputs, xor_context.expand(4, -1)))
            and_pred = torch.sigmoid(model(test_inputs, and_context.expand(4, -1)))
            
            xor_true = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
            and_true = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
            
            xor_acc = ((xor_pred > 0.5).float() == xor_true).float().mean()
            and_acc = ((and_pred > 0.5).float() == and_true).float().mean()
            
            return xor_acc.item(), and_acc.item(), xor_pred, and_pred
    
    print("\nINITIAL PERFORMANCE (before training):")
    xor_acc, and_acc, xor_pred, and_pred = test_performance()
    print(f"XOR Accuracy: {xor_acc:.3f}, AND Accuracy: {and_acc:.3f}")
    print("XOR predictions:", [f"{x:.3f}" for x in xor_pred.flatten().tolist()])
    print("AND predictions:", [f"{x:.3f}" for x in and_pred.flatten().tolist()])
    
    # Check initial gating
    print("\nINITIAL GATING:")
    model.configure_context(xor_context)
    xor_gates = model.gating_networks[0](test_inputs[:1])
    model.configure_context(and_context)
    and_gates = model.gating_networks[0](test_inputs[:1])
    print(f"XOR gates: {xor_gates.flatten()[:8]}")
    print(f"AND gates: {and_gates.flatten()[:8]}")
    print(f"Initial gate difference: {(xor_gates - and_gates).abs().mean().item():.6f}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Lower learning rate
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_size = 16  # Smaller batch size
    epochs = 300
    
    print(f"\nStarting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle data
        indices = torch.randperm(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        contexts_shuffled = contexts[indices]
        
        # Mini-batch training
        for i in range(0, X.shape[0], batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            batch_contexts = contexts_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            
            outputs = model(batch_X, batch_contexts)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        # Print progress every 30 epochs
        if (epoch + 1) % 30 == 0:
            avg_loss = total_loss / (X.shape[0] // batch_size)
            accuracy = correct / total
            
            # Check hypernetwork differentiation during training
            model.eval()
            with torch.no_grad():
                hyper_xor = model.hypernetwork(xor_context)
                hyper_and = model.hypernetwork(and_context)
                hyper_diff = (hyper_xor - hyper_and).abs().mean().item()
            model.train()
            
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, Hyper diff = {hyper_diff:.4f}")
    
    print("\nFINAL PERFORMANCE (after training):")
    xor_acc, and_acc, xor_pred, and_pred = test_performance()
    print(f"XOR Accuracy: {xor_acc:.3f}, AND Accuracy: {and_acc:.3f}")
    print("XOR predictions:", [f"{x:.3f}" for x in xor_pred.flatten().tolist()])
    print("AND predictions:", [f"{x:.3f}" for x in and_pred.flatten().tolist()])
    
    # Test context switching with same input
    print("\nCONTEXT SWITCHING DEMONSTRATION:")
    test_input = torch.tensor([[1, 1]], dtype=torch.float32)  # Input [1,1]
    
    xor_out = torch.sigmoid(model(test_input, xor_context))
    and_out = torch.sigmoid(model(test_input, and_context))
    
    print(f"Input [1,1] with XOR context: {xor_out.item():.4f} (should be ~0.0)")
    print(f"Input [1,1] with AND context: {and_out.item():.4f} (should be ~1.0)")
    print(f"Context difference: {abs(xor_out.item() - and_out.item()):.4f}")
    
    # Analyze gate differences after training
    print("\nFINAL GATE ANALYSIS:")
    model.configure_context(xor_context)
    xor_gates = model.gating_networks[0](test_input)
    model.configure_context(and_context)  
    and_gates = model.gating_networks[0](test_input)
    
    print(f"XOR gates: {xor_gates.flatten()}")
    print(f"AND gates: {and_gates.flatten()}")
    print(f"Final gate difference: {(xor_gates - and_gates).abs().mean().item():.4f}")
    
    return model

# Add this to the main execution
if __name__ == "__main__":
    # Original architecture test
    input_size = 10
    context_size = 3
    main_layers = [(8, 4, 6), (4, 3, 8)]  # (n_neurons, n_branches, branch_size)
    output_size = 2
    
    # Create model
    model = HypernetworkGatedDANN(
        input_size=input_size,
        context_size=context_size,
        main_layers=main_layers,
        output_size=output_size
    )
    
    # Print architecture info
    info = model.get_architecture_info()
    print("Architecture Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 5
    x = torch.randn(batch_size, input_size)
    context = torch.randn(batch_size, context_size)
    
    # Forward pass
    output = model(x, context)
    print(f"\nInput shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test different contexts with diagnostics
    context1 = torch.tensor([[1.0, 0.0, 0.0]]).expand(batch_size, -1)  # Task 1
    context2 = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_size, -1)  # Task 2
    
    # Test with same input but different contexts
    test_x = torch.randn(1, input_size)  # Single sample
    test_context1 = torch.tensor([[1.0, 0.0, 0.0]])  # Task 1
    test_context2 = torch.tensor([[0.0, 1.0, 0.0]])  # Task 2
    
    # Test different contexts with diagnostics - using more distinct contexts
    test_context1 = torch.tensor([[1.0, 0.0, 0.0]])  # Task 1
    test_context2 = torch.tensor([[0.0, 0.0, 1.0]])  # Task 3 (more different)
    
    # Check what the hypernetwork produces for different contexts
    print(f"\nHypernetwork diagnostics:")
    print(f"Context 1: {test_context1.flatten()}")
    print(f"Context 2: {test_context2.flatten()}")
    
    hyper_out1 = model.hypernetwork(test_context1)
    hyper_out2 = model.hypernetwork(test_context2)
    print(f"Hypernetwork output 1 (first 10 values): {hyper_out1[0, :10]}")
    print(f"Hypernetwork output 2 (first 10 values): {hyper_out2[0, :10]}")
    print(f"Hypernetwork output difference: {(hyper_out1 - hyper_out2).abs().mean().item():.6f}")
    
    # If difference is still too small, let's check the hypernetwork internals
    if (hyper_out1 - hyper_out2).abs().mean().item() < 0.01:
        print("\nHypernetwork internal analysis:")
        with torch.no_grad():
            # Check first layer output
            first_layer = model.hypernetwork.network[0]
            h1_layer1 = first_layer(test_context1)
            h2_layer1 = first_layer(test_context2)
            print(f"First layer difference: {(h1_layer1 - h2_layer1).abs().mean().item():.6f}")
            
            # Check if contexts are actually different
            context_diff = (test_context1 - test_context2).abs().mean().item()
            print(f"Input context difference: {context_diff:.6f}")
    
    # Check gate values
    model.configure_context(test_context1)
    gates1 = model.gating_networks[0](test_x)
    model.configure_context(test_context2)
    gates2 = model.gating_networks[0](test_x)
    print(f"Gate values 1 (first 10): {gates1[0, :10]}")
    print(f"Gate values 2 (first 10): {gates2[0, :10]}")
    print(f"Gate difference: {(gates1 - gates2).abs().mean().item():.6f}")
    
    output1 = model(test_x, test_context1)
    output2 = model(test_x, test_context2)
    
    print(f"\nSame input X with different contexts:")
    print(f"Context 1 output: {output1.flatten()}")
    print(f"Context 2 output: {output2.flatten()}")
    print(f"Output difference: {(output1 - output2).abs().mean().item():.6f}")
    
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters())}")
    print("Model is ready for training with standard PyTorch optimizers!")
    
    # Run the multi-task training demonstration
    print("\n" + "="*80)
    print("MULTI-TASK TRAINING DEMONSTRATION")
    print("="*80)
    trained_model = test_training_multitask()
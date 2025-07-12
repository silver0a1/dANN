import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import math
from scipy import stats
from sklearn.metrics import r2_score

class MathematicalVerification:
    """Comprehensive mathematical verification suite for the Hypernetwork-Gated DANN"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

        # Define the mathematical functions we're testing (torch tensor versions)
        self.functions = {
            'polynomial': self._polynomial_func,
            'trigonometric': self._trigonometric_func,
            'step': self._step_func,
            'exponential': self._exponential_func
        }

        # Context vectors for each function
        self.contexts = {
            'polynomial': torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            'trigonometric': torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device),
            'step': torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device),
            'exponential': torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        }

    def _polynomial_func(self, x):
        """Polynomial function: x^2"""
        return x ** 2

    def _trigonometric_func(self, x):
        """Trigonometric function: sin(2πx/4)"""
        if torch.is_tensor(x):
            return torch.sin(2 * math.pi * x / 4)
        else:
            return np.sin(2 * math.pi * x / 4)

    def _step_func(self, x):
        """Step function: 1 if x > 0 else 0"""
        if torch.is_tensor(x):
            return (x > 0).float()
        elif isinstance(x, np.ndarray):
            return (x > 0).astype(float)
        else:
            # Handle scalar case
            return 1.0 if x > 0 else 0.0

    def _exponential_func(self, x):
        """Exponential function: exp(x/2)"""
        if torch.is_tensor(x):
            return torch.exp(x / 2)
        else:
            return np.exp(x / 2)

    def verify_function_approximation(self, n_test_points=1000, x_range=(-2.0, 2.0)):
        """Verify that the model accurately approximates each mathematical function"""
        results = {}
        x_test = torch.linspace(x_range[0], x_range[1], n_test_points, device=self.device).unsqueeze(1)

        self.model.eval()
        with torch.no_grad():
            for func_name, func in self.functions.items():
                context = self.contexts[func_name].expand(n_test_points, -1)

                # Model predictions
                y_pred = self.model(x_test, context).squeeze()

                # True values
                y_true = func(x_test.squeeze())

                # Calculate metrics
                mse = F.mse_loss(y_pred, y_true).item()
                mae = F.l1_loss(y_pred, y_true).item()

                # R-squared score
                y_true_np = y_true.cpu().numpy()
                y_pred_np = y_pred.cpu().numpy()
                r2 = r2_score(y_true_np, y_pred_np)

                # Maximum absolute error
                max_error = torch.max(torch.abs(y_pred - y_true)).item()

                # Relative error (avoiding division by zero)
                mask = torch.abs(y_true) > 1e-6
                relative_errors = torch.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
                mean_relative_error = relative_errors.mean().item() if mask.any() else 0.0

                results[func_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'max_error': max_error,
                    'mean_relative_error': mean_relative_error
                }

        return results

    def verify_context_separation(self, test_points=None):
        """Verify that different contexts produce appropriately different outputs"""
        if test_points is None:
            test_points = torch.tensor([[-1.5], [0.0], [0.5], [1.0], [1.5]], device=self.device)

        self.model.eval()
        results = {}

        with torch.no_grad():
            for x_val in test_points:
                x_expanded = x_val.expand(1, -1)
                outputs = {}

                for func_name, context in self.contexts.items():
                    output = self.model(x_expanded, context).item()
                    x_scalar = x_val.item()
                    true_value = self.functions[func_name](x_scalar)
                    true_scalar = true_value if isinstance(true_value, (int, float)) else true_value.item()
                    outputs[func_name] = {
                        'predicted': output,
                        'true': true_scalar,
                        'error': abs(output - true_scalar)
                    }

                results[f'x={x_val.item():.2f}'] = outputs

        return results

    def verify_mathematical_properties(self):
        """Verify specific mathematical properties of each function"""
        results = {}

        self.model.eval()
        with torch.no_grad():
            # Test polynomial properties
            results['polynomial'] = self._verify_polynomial_properties()

            # Test trigonometric properties
            results['trigonometric'] = self._verify_trigonometric_properties()

            # Test step function properties
            results['step'] = self._verify_step_properties()

            # Test exponential properties
            results['exponential'] = self._verify_exponential_properties()

        return results

    def _verify_polynomial_properties(self):
        """Verify quadratic properties: f(x) = x^2"""
        props = {}
        context = self.contexts['polynomial']

        # Property 1: Symmetry f(x) = f(-x)
        test_x = torch.linspace(0.1, 2.0, 20, device=self.device)
        pos_x = test_x.unsqueeze(1)
        neg_x = -test_x.unsqueeze(1)

        pos_out = self.model(pos_x, context.expand(20, -1))
        neg_out = self.model(neg_x, context.expand(20, -1))

        symmetry_error = F.mse_loss(pos_out, neg_out).item()
        props['symmetry_error'] = symmetry_error
        props['is_symmetric'] = symmetry_error < 0.01

        # Property 2: Minimum at x=0
        x_around_zero = torch.linspace(-0.5, 0.5, 50, device=self.device).unsqueeze(1)
        y_around_zero = self.model(x_around_zero, context.expand(50, -1)).squeeze()
        min_idx = torch.argmin(y_around_zero)
        min_x = x_around_zero[min_idx].item()
        props['minimum_at'] = min_x
        props['minimum_error'] = abs(min_x)

        # Property 3: Quadratic growth rate
        x_vals = torch.tensor([[1.0], [2.0], [3.0]], device=self.device)
        y_vals = self.model(x_vals, context.expand(3, -1)).squeeze()

        # Check if y ≈ x^2
        expected = x_vals.squeeze() ** 2
        growth_error = F.mse_loss(y_vals, expected).item()
        props['quadratic_growth_error'] = growth_error

        return props

    def _verify_trigonometric_properties(self):
        """Verify sine properties: f(x) = sin(2πx/4)"""
        props = {}
        context = self.contexts['trigonometric']

        # Property 1: Periodicity with period 4
        x_base = torch.linspace(-2, 2, 50, device=self.device).unsqueeze(1)
        x_shifted = x_base + 4.0  # One period shift

        y_base = self.model(x_base, context.expand(50, -1))
        y_shifted = self.model(x_shifted, context.expand(50, -1))

        periodicity_error = F.mse_loss(y_base, y_shifted).item()
        props['periodicity_error'] = periodicity_error
        props['is_periodic'] = periodicity_error < 0.01

        # Property 2: Zero crossings at x = 0, 2, 4, -2, -4
        zero_points = torch.tensor([[0.0], [2.0], [-2.0]], device=self.device)
        y_at_zeros = self.model(zero_points, context.expand(3, -1)).squeeze()
        zero_crossing_error = torch.abs(y_at_zeros).mean().item()
        props['zero_crossing_error'] = zero_crossing_error

        # Property 3: Amplitude ≈ 1
        x_full = torch.linspace(-4, 4, 200, device=self.device).unsqueeze(1)
        y_full = self.model(x_full, context.expand(200, -1)).squeeze()
        amplitude = (y_full.max() - y_full.min()).item() / 2
        props['amplitude'] = amplitude
        props['amplitude_error'] = abs(amplitude - 1.0)

        return props

    def _verify_step_properties(self):
        """Verify step function properties: f(x) = 1 if x > 0 else 0"""
        props = {}
        context = self.contexts['step']

        # Property 1: Output ≈ 0 for x < 0
        x_negative = torch.linspace(-2, -0.1, 20, device=self.device).unsqueeze(1)
        y_negative = self.model(x_negative, context.expand(20, -1)).squeeze()
        negative_error = torch.abs(y_negative).mean().item()
        props['negative_side_error'] = negative_error

        # Property 2: Output ≈ 1 for x > 0
        x_positive = torch.linspace(0.1, 2, 20, device=self.device).unsqueeze(1)
        y_positive = self.model(x_positive, context.expand(20, -1)).squeeze()
        positive_error = torch.abs(y_positive - 1.0).mean().item()
        props['positive_side_error'] = positive_error

        # Property 3: Sharp transition at x = 0
        x_transition = torch.linspace(-0.1, 0.1, 100, device=self.device).unsqueeze(1)
        y_transition = self.model(x_transition, context.expand(100, -1)).squeeze()

        # Find steepest gradient
        gradients = torch.diff(y_transition) / torch.diff(x_transition.squeeze())
        max_gradient = torch.max(torch.abs(gradients)).item()
        props['transition_sharpness'] = max_gradient
        props['is_sharp'] = max_gradient > 5.0  # Should be very steep

        return props

    def _verify_exponential_properties(self):
        """Verify exponential properties: f(x) = exp(x/2)"""
        props = {}
        context = self.contexts['exponential']

        # Property 1: f(0) = 1
        x_zero = torch.tensor([[0.0]], device=self.device)
        y_zero = self.model(x_zero, context).item()
        props['value_at_zero'] = y_zero
        props['zero_error'] = abs(y_zero - 1.0)

        # Property 2: Exponential growth rate - f(x+2) / f(x) ≈ e
        x_vals = torch.linspace(-1, 1, 10, device=self.device).unsqueeze(1)
        y_vals = self.model(x_vals, context.expand(10, -1)).squeeze()

        x_shifted = x_vals + 2.0
        y_shifted = self.model(x_shifted, context.expand(10, -1)).squeeze()

        # Avoid division by very small numbers
        mask = y_vals.abs() > 0.1
        if mask.any():
            growth_ratios = y_shifted[mask] / y_vals[mask]
            expected_ratio = math.e  # Since exp((x+2)/2) / exp(x/2) = exp(1) = e
            growth_error = torch.abs(growth_ratios - expected_ratio).mean().item()
            props['growth_rate_error'] = growth_error
            props['mean_growth_ratio'] = growth_ratios.mean().item()
        else:
            props['growth_rate_error'] = float('inf')
            props['mean_growth_ratio'] = 0.0

        # Property 3: Always positive
        x_test = torch.linspace(-3, 3, 100, device=self.device).unsqueeze(1)
        y_test = self.model(x_test, context.expand(100, -1)).squeeze()
        min_value = y_test.min().item()
        props['minimum_value'] = min_value
        props['is_always_positive'] = min_value > -0.01

        return props

    def verify_gradient_flow(self):
        """Verify that gradients flow properly through all components"""
        results = {}

        # Create a simple test case
        x_test = torch.tensor([[1.0]], device=self.device, requires_grad=True)
        context_test = self.contexts['polynomial']

        # Forward pass
        output = self.model(x_test, context_test)

        # Backward pass
        self.model.zero_grad()
        output.backward()

        # Check gradients in different components
        results['input_gradient'] = x_test.grad.abs().item() if x_test.grad is not None else 0.0

        # Check main network gradients
        main_grad_norms = []
        for param in self.model.main_network.parameters():
            if param.grad is not None:
                main_grad_norms.append(param.grad.norm().item())
        results['main_network_grad_norm'] = np.mean(main_grad_norms) if main_grad_norms else 0.0

        # Check hypernetwork gradients
        hyper_grad_norms = []
        for param in self.model.hypernetwork.parameters():
            if param.grad is not None:
                hyper_grad_norms.append(param.grad.norm().item())
        results['hypernetwork_grad_norm'] = np.mean(hyper_grad_norms) if hyper_grad_norms else 0.0

        results['gradient_flow_healthy'] = (
            results['main_network_grad_norm'] > 1e-6 and
            results['hypernetwork_grad_norm'] > 1e-6
        )

        return results

    def verify_gate_differentiation(self):
        """Verify that different contexts produce different gating patterns"""
        results = {}

        # Test point
        x_test = torch.tensor([[0.0]], device=self.device)

        self.model.eval()
        with torch.no_grad():
            gate_patterns = {}

            for func_name, context in self.contexts.items():
                # Get hypernetwork output for this context
                hyper_output = self.model.hypernetwork(context)

                # Configure gating networks and get gate values
                idx = 0
                gates = []
                for gating_net in self.model.gating_networks:
                    param_count = gating_net.total_params
                    gating_weights = hyper_output[0, idx:idx + param_count]
                    gating_net.set_hypernetwork_outputs(gating_weights)
                    gate_values = gating_net(x_test)
                    gates.append(gate_values.flatten())
                    idx += param_count

                gate_patterns[func_name] = torch.cat(gates)

            # Calculate pairwise distances between gate patterns
            pattern_list = list(gate_patterns.values())
            n_patterns = len(pattern_list)
            distances = torch.zeros(n_patterns, n_patterns)

            for i in range(n_patterns):
                for j in range(n_patterns):
                    distances[i, j] = F.mse_loss(pattern_list[i], pattern_list[j]).item()

            results['gate_distance_matrix'] = distances.numpy()
            results['mean_pairwise_distance'] = distances[distances > 0].mean().item()
            results['min_pairwise_distance'] = distances[distances > 0].min().item()

            # Check if all patterns are sufficiently different
            threshold = 0.01
            results['all_patterns_different'] = results['min_pairwise_distance'] > threshold

        return results

    def generate_verification_report(self, save_plots=True):
        """Generate a comprehensive verification report"""
        print("=" * 80)
        print("MATHEMATICAL VERIFICATION REPORT")
        print("=" * 80)

        # 1. Function Approximation Quality
        print("\n1. FUNCTION APPROXIMATION QUALITY")
        print("-" * 40)
        approx_results = self.verify_function_approximation()

        for func_name, metrics in approx_results.items():
            print(f"\n{func_name.upper()}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  R²:  {metrics['r2']:.6f}")
            print(f"  Max Error: {metrics['max_error']:.6f}")
            print(f"  Mean Relative Error: {metrics['mean_relative_error']:.4%}")

        # 2. Context Separation
        print("\n\n2. CONTEXT SEPARATION VERIFICATION")
        print("-" * 40)
        separation_results = self.verify_context_separation()

        for x_point, outputs in separation_results.items():
            print(f"\n{x_point}:")
            for func_name, values in outputs.items():
                print(f"  {func_name:15s}: pred={values['predicted']:7.4f}, "
                      f"true={values['true']:7.4f}, error={values['error']:7.4f}")

        # 3. Mathematical Properties
        print("\n\n3. MATHEMATICAL PROPERTIES VERIFICATION")
        print("-" * 40)
        prop_results = self.verify_mathematical_properties()

        print("\nPOLYNOMIAL (x²):")
        print(f"  Symmetric: {prop_results['polynomial']['is_symmetric']} "
              f"(error: {prop_results['polynomial']['symmetry_error']:.6f})")
        print(f"  Minimum at x = {prop_results['polynomial']['minimum_at']:.4f} "
              f"(error: {prop_results['polynomial']['minimum_error']:.6f})")
        print(f"  Quadratic growth error: {prop_results['polynomial']['quadratic_growth_error']:.6f}")

        print("\nTRIGONOMETRIC (sin(2πx/4)):")
        print(f"  Periodic: {prop_results['trigonometric']['is_periodic']} "
              f"(error: {prop_results['trigonometric']['periodicity_error']:.6f})")
        print(f"  Zero crossing error: {prop_results['trigonometric']['zero_crossing_error']:.6f}")
        print(f"  Amplitude: {prop_results['trigonometric']['amplitude']:.4f} "
              f"(error: {prop_results['trigonometric']['amplitude_error']:.6f})")

        print("\nSTEP FUNCTION:")
        print(f"  Negative side error: {prop_results['step']['negative_side_error']:.6f}")
        print(f"  Positive side error: {prop_results['step']['positive_side_error']:.6f}")
        print(f"  Sharp transition: {prop_results['step']['is_sharp']} "
              f"(sharpness: {prop_results['step']['transition_sharpness']:.2f})")

        print("\nEXPONENTIAL (exp(x/2)):")
        print(f"  f(0) = {prop_results['exponential']['value_at_zero']:.4f} "
              f"(error: {prop_results['exponential']['zero_error']:.6f})")
        print(f"  Growth rate error: {prop_results['exponential']['growth_rate_error']:.6f}")
        print(f"  Always positive: {prop_results['exponential']['is_always_positive']} "
              f"(min: {prop_results['exponential']['minimum_value']:.6f})")

        # 4. Gradient Flow
        print("\n\n4. GRADIENT FLOW VERIFICATION")
        print("-" * 40)
        grad_results = self.verify_gradient_flow()
        print(f"  Input gradient magnitude: {grad_results['input_gradient']:.6f}")
        print(f"  Main network gradient norm: {grad_results['main_network_grad_norm']:.6f}")
        print(f"  Hypernetwork gradient norm: {grad_results['hypernetwork_grad_norm']:.6f}")
        print(f"  Gradient flow healthy: {grad_results['gradient_flow_healthy']}")

        # 5. Gate Differentiation
        print("\n\n5. GATE DIFFERENTIATION VERIFICATION")
        print("-" * 40)
        gate_results = self.verify_gate_differentiation()
        print(f"  Mean pairwise gate distance: {gate_results['mean_pairwise_distance']:.6f}")
        print(f"  Min pairwise gate distance: {gate_results['min_pairwise_distance']:.6f}")
        print(f"  All patterns sufficiently different: {gate_results['all_patterns_different']}")

        # Overall assessment
        print("\n\n6. OVERALL ASSESSMENT")
        print("-" * 40)

        # Check if all functions are well approximated
        all_r2 = [metrics['r2'] for metrics in approx_results.values()]
        all_good_fit = all(r2 > 0.95 for r2 in all_r2)

        # Check if properties are satisfied
        props_satisfied = (
            prop_results['polynomial']['is_symmetric'] and
            prop_results['trigonometric']['is_periodic'] and
            prop_results['step']['is_sharp'] and
            prop_results['exponential']['is_always_positive']
        )

        print(f"  All functions well approximated (R² > 0.95): {all_good_fit}")
        print(f"  Mathematical properties satisfied: {props_satisfied}")
        print(f"  Gradient flow healthy: {grad_results['gradient_flow_healthy']}")
        print(f"  Gate patterns differentiated: {gate_results['all_patterns_different']}")

        success = all_good_fit and props_satisfied and grad_results['gradient_flow_healthy'] and gate_results['all_patterns_different']
        print(f"\n  ✓ VERIFICATION {'PASSED' if success else 'FAILED'}")

        if save_plots:
            self._generate_verification_plots()

        return {
            'approximation': approx_results,
            'separation': separation_results,
            'properties': prop_results,
            'gradients': grad_results,
            'gates': gate_results,
            'success': success
        }

    def _generate_verification_plots(self):
        """Generate visualization plots for verification"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        x_plot = torch.linspace(-2, 2, 200, device=self.device).unsqueeze(1)

        self.model.eval()
        with torch.no_grad():
            for idx, (func_name, func) in enumerate(self.functions.items()):
                ax = axes[idx]

                # True function
                x_numpy = x_plot.squeeze().cpu().numpy()
                y_true = func(x_numpy)

                # Model prediction
                context = self.contexts[func_name].expand(200, -1)
                y_pred = self.model(x_plot, context).squeeze().cpu().numpy()

                # Plot
                ax.plot(x_numpy, y_true, 'b-', label='True', linewidth=2)
                ax.plot(x_numpy, y_pred, 'r--', label='Predicted', linewidth=2)
                ax.set_title(f'{func_name.capitalize()} Function')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add error text
                mse = np.mean((y_true - y_pred) ** 2)
                ax.text(0.05, 0.95, f'MSE: {mse:.6f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Function Approximation Verification', fontsize=16)
        plt.tight_layout()
        plt.savefig('verification_plots.png', dpi=150)
        plt.close()

        # Gate pattern heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get gate patterns for visualization
        x_test = torch.tensor([[0.0]], device=self.device)
        gate_matrix = []

        with torch.no_grad():
            for func_name, context in self.contexts.items():
                hyper_output = self.model.hypernetwork(context)
                idx = 0
                gates = []
                for gating_net in self.model.gating_networks:
                    param_count = gating_net.total_params
                    gating_weights = hyper_output[0, idx:idx + param_count]
                    gating_net.set_hypernetwork_outputs(gating_weights)
                    gate_values = gating_net(x_test)
                    gates.extend(gate_values.flatten().cpu().numpy())
                    idx += param_count
                gate_matrix.append(gates)

        gate_matrix = np.array(gate_matrix)

        im = ax.imshow(gate_matrix, aspect='auto', cmap='viridis')
        ax.set_yticks(range(4))
        ax.set_yticklabels(['Polynomial', 'Trigonometric', 'Step', 'Exponential'])
        ax.set_xlabel('Gate Index')
        ax.set_title('Gate Activation Patterns by Function Context')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('gate_patterns.png', dpi=150)
        plt.close()

        print("\n  Plots saved: verification_plots.png, gate_patterns.png")


# Example usage with your trained model
def run_verification(trained_model):
    """Run the complete verification suite on a trained model"""
    verifier = MathematicalVerification(trained_model)
    results = verifier.generate_verification_report(save_plots=True)
    return results
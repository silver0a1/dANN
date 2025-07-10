# Run enhanced training
model, results, history = enhanced_test_mathematical_reasoning()

# If verification still fails, analyze what's going wrong
if not results['success']:
    issues = analyze_verification_failures(results)
#begin file
def enhanced_test_mathematical_reasoning():
    """Enhanced training with improvements for better verification success"""
    print("="*70)
    print("ENHANCED MULTI-FUNCTION MATHEMATICAL REASONING")
    print("="*70)

    # Create model with slightly larger architecture
    model = HypernetworkGatedDANN(
        input_size=1,
        context_size=4,
        main_layers=[(8, 4, 6), (6, 3, 4)],  # Two layers for more capacity
        output_size=1,
        hypernetwork_hidden=[32, 64, 32],  # Deeper hypernetwork
        gating_hidden=16  # Larger gating hidden size
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create larger, more diverse dataset
    X, Y, contexts, function_types = create_mathematical_functions_dataset(500)  # More samples
    print(f"Dataset created: {X.shape[0]} samples across 4 mathematical functions")

    # Enhanced training setup
    main_params = list(model.main_network.parameters()) + list(model.output_layer.parameters())
    hyper_params = list(model.hypernetwork.parameters())

    # Adaptive learning rates with warmup
    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': 0.001, 'weight_decay': 1e-5},
        {'params': hyper_params, 'lr': 0.02, 'weight_decay': 1e-6}  # Higher LR for hypernetwork
    ], betas=(0.9, 0.999))

    # Cosine annealing scheduler with warm restarts
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    # Multiple loss functions for different function types
    mse_loss = nn.MSELoss()
    smooth_l1_loss = nn.SmoothL1Loss()  # Better for step function
    
    batch_size = 64
    epochs = 400
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 100

    print(f"\nStarting enhanced training for {epochs} epochs...")
    
    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'poly_loss': [],
        'trig_loss': [],
        'step_loss': [],
        'exp_loss': []
    }

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        function_losses = {fn: 0 for fn in ['polynomial', 'trigonometric', 'step', 'exponential']}
        function_counts = {fn: 0 for fn in ['polynomial', 'trigonometric', 'step', 'exponential']}

        # Shuffle data
        indices = torch.randperm(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        contexts_shuffled = contexts[indices]
        types_shuffled = [function_types[i] for i in indices]

        # Mini-batch training
        for i in range(0, X.shape[0], batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_Y = Y_shuffled[i:i+batch_size]
            batch_contexts = contexts_shuffled[i:i+batch_size]
            batch_types = types_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X, batch_contexts)
            
            # Calculate loss with function-specific handling
            batch_loss = 0
            for j in range(len(batch_X)):
                if batch_types[j] == 'step':
                    # Use smooth L1 loss for step function (more robust to outliers)
                    loss = smooth_l1_loss(outputs[j], batch_Y[j])
                else:
                    # Use MSE for continuous functions
                    loss = mse_loss(outputs[j], batch_Y[j])
                
                batch_loss += loss
                function_losses[batch_types[j]] += loss.item()
                function_counts[batch_types[j]] += 1
            
            batch_loss = batch_loss / len(batch_X)
            
            # Add regularization to encourage gate diversity
            if epoch > 20:  # After initial training
                # Get gate patterns for different contexts in the batch
                unique_contexts = torch.unique(batch_contexts, dim=0)
                if len(unique_contexts) > 1:
                    gate_patterns = []
                    for ctx in unique_contexts:
                        hyper_out = model.hypernetwork(ctx.unsqueeze(0))
                        gate_patterns.append(hyper_out)
                    
                    # Encourage diversity between gate patterns
                    gate_diversity_loss = 0
                    for i in range(len(gate_patterns)):
                        for j in range(i+1, len(gate_patterns)):
                            similarity = F.cosine_similarity(
                                gate_patterns[i].flatten(), 
                                gate_patterns[j].flatten(), 
                                dim=0
                            )
                            gate_diversity_loss += similarity ** 2
                    
                    # Add small diversity penalty
                    batch_loss += 0.01 * gate_diversity_loss

            batch_loss.backward()
            
            # Gradient clipping with different norms for different components
            torch.nn.utils.clip_grad_norm_(main_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(hyper_params, max_norm=2.0)
            
            optimizer.step()
            total_loss += batch_loss.item()

        # Calculate average losses
        avg_total_loss = total_loss / (X.shape[0] // batch_size)
        avg_function_losses = {
            fn: function_losses[fn] / function_counts[fn] if function_counts[fn] > 0 else 0
            for fn in function_losses
        }
        
        # Update learning rate
        scheduler_main.step()
        
        # Early stopping check
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Detailed progress reporting
        if (epoch + 1) % 10 == 0:
            # Check current performance
            model.eval()
            with torch.no_grad():
                # Quick verification check
                verifier = MathematicalVerification(model)
                approx_results = verifier.verify_function_approximation(n_test_points=100)
                gate_results = verifier.verify_gate_differentiation()
                
                avg_r2 = np.mean([res['r2'] for res in approx_results.values()])
                gate_diff = gate_results['mean_pairwise_distance']
            
            model.train()
            
            print(f"Epoch {epoch+1:3d}: Loss = {avg_total_loss:.4f}, "
                  f"R² = {avg_r2:.4f}, Gate diff = {gate_diff:.4f}")
            print(f"  Function losses - Poly: {avg_function_losses['polynomial']:.4f}, "
                  f"Trig: {avg_function_losses['trigonometric']:.4f}, "
                  f"Step: {avg_function_losses['step']:.4f}, "
                  f"Exp: {avg_function_losses['exponential']:.4f}")
        
        # Store history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_total_loss)
        history['poly_loss'].append(avg_function_losses['polynomial'])
        history['trig_loss'].append(avg_function_losses['trigonometric'])
        history['step_loss'].append(avg_function_losses['step'])
        history['exp_loss'].append(avg_function_losses['exponential'])
        
        # Adaptive early stopping
        if patience_counter > max_patience and epoch > 100:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Final verification
    print("\nRunning final verification...")
    verifier = MathematicalVerification(model)
    final_results = verifier.generate_verification_report(save_plots=True)
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Total loss
    ax1.plot(history['epoch'], history['total_loss'], 'b-', label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Function-specific losses
    ax2.plot(history['epoch'], history['poly_loss'], label='Polynomial', linewidth=2)
    ax2.plot(history['epoch'], history['trig_loss'], label='Trigonometric', linewidth=2)
    ax2.plot(history['epoch'], history['step_loss'], label='Step', linewidth=2)
    ax2.plot(history['epoch'], history['exp_loss'], label='Exponential', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Function-Specific Losses')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    
    return model, final_results, history


def analyze_verification_failures(verification_results):
    """Analyze which parts of verification are failing"""
    print("\n" + "="*60)
    print("VERIFICATION FAILURE ANALYSIS")
    print("="*60)
    
    # Check approximation quality
    print("\n1. Approximation Quality:")
    for func, metrics in verification_results['approximation'].items():
        status = "✓" if metrics['r2'] > 0.95 else "✗"
        print(f"  {status} {func}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.6f}")
    
    # Check mathematical properties
    print("\n2. Mathematical Properties:")
    props = verification_results['properties']
    
    print(f"  {'✓' if props['polynomial']['is_symmetric'] else '✗'} Polynomial symmetry")
    print(f"  {'✓' if props['trigonometric']['is_periodic'] else '✗'} Trigonometric periodicity")
    print(f"  {'✓' if props['step']['is_sharp'] else '✗'} Step function sharpness")
    print(f"  {'✓' if props['exponential']['is_always_positive'] else '✗'} Exponential positivity")
    
    # Check gradient flow
    print("\n3. Gradient Flow:")
    grads = verification_results['gradients']
    print(f"  {'✓' if grads['gradient_flow_healthy'] else '✗'} Healthy gradient flow")
    print(f"    Main network grad: {grads['main_network_grad_norm']:.6f}")
    print(f"    Hypernetwork grad: {grads['hypernetwork_grad_norm']:.6f}")
    
    # Check gate differentiation
    print("\n4. Gate Differentiation:")
    gates = verification_results['gates']
    print(f"  {'✓' if gates['all_patterns_different'] else '✗'} All patterns different")
    print(f"    Mean pairwise distance: {gates['mean_pairwise_distance']:.6f}")
    print(f"    Min pairwise distance: {gates['min_pairwise_distance']:.6f}")
    
    # Identify main failure points
    print("\n5. Main Issues:")
    issues = []
    
    for func, metrics in verification_results['approximation'].items():
        if metrics['r2'] < 0.95:
            issues.append(f"Poor {func} approximation (R²={metrics['r2']:.3f})")
    
    if not gates['all_patterns_different']:
        issues.append("Insufficient gate pattern differentiation")
    
    if not grads['gradient_flow_healthy']:
        issues.append("Unhealthy gradient flow")
    
    if not issues:
        issues.append("No major issues detected!")
    
    for issue in issues:
        print(f"  - {issue}")
    
    return issues
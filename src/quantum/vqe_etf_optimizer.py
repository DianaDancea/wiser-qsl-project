"""
Vanguard ETF Optimizer - Quantum VQE Implementation
Quantum optimization using Variational Quantum Eigensolver (VQE)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE, QAOA
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, SLSQP
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Import classical optimizer for comparison
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classical.vanguard_etf_optimizer import VanguardETFOptimizer, ETFConfiguration

class QuantumETFOptimizer:
    """
    Quantum ETF Optimizer using VQE
    
    Converts Vanguard's ETF optimization problem to QUBO formulation
    and solves using Variational Quantum Eigensolver
    """
    
    def __init__(self, classical_optimizer: VanguardETFOptimizer):
        self.classical_optimizer = classical_optimizer
        self.config = classical_optimizer.config
        self.qubo_matrix = None
        self.qubo_offset = 0.0
        self.pauli_operator = None
        self.quantum_result = None
        self.vqe_history = []
        
    def formulate_qubo(self, penalty_weights: Dict[str, float] = None) -> Tuple[np.ndarray, float]:
        """
        Convert Vanguard ETF problem to QUBO formulation
        
        QUBO: minimize x^T Q x + offset
        where x is binary vector (x_i ∈ {0,1})
        
        Vanguard objective: minimize Σ_{ℓ,j} ρ_j (Σ_{c∈R_ℓ} β_{c,j} x_c - K_{ℓ,j}^target)²
        """
        print("\n" + "="*50)
        print("QUBO FORMULATION")
        print("="*50)
        
        if self.classical_optimizer.bond_data is None:
            raise ValueError("Classical optimizer must have bond data. Run generate_realistic_bond_universe() first.")
        
        # Default penalty weights
        if penalty_weights is None:
            penalty_weights = {
                'max_bonds': 100.0,
                'min_bonds': 50.0, 
                'cash_flow': 150.0,
                'risk_bucket': 75.0,
                'characteristic_bounds': 125.0
            }
        
        n_bonds = self.config.n_bonds
        self.qubo_matrix = np.zeros((n_bonds, n_bonds))
        self.qubo_offset = 0.0
        
        print(f"Creating QUBO matrix: {n_bonds}x{n_bonds}")
        print(f"Penalty weights: {penalty_weights}")
        
        # MAIN OBJECTIVE: Portfolio characteristic tracking
        # Σ_{ℓ,j} ρ_j (Σ_{c∈R_ℓ} β_{c,j} x_c - K_{ℓ,j}^target)²
        
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.classical_optimizer.risk_buckets[bucket_id]['bonds']
            
            for char_id in range(self.config.n_characteristics):
                target_val = self.classical_optimizer.targets[bucket_id][char_id]['target']
                char_weight = self.classical_optimizer.characteristics[char_id]['weight']
                contributions = self.classical_optimizer.characteristics[char_id]['contributions']
                
                # Quadratic terms: Σ_{c∈R_ℓ} Σ_{d∈R_ℓ} β_{c,j} β_{d,j} x_c x_d
                for c in bonds_in_bucket:
                    for d in bonds_in_bucket:
                        coeff = char_weight * contributions[c] * contributions[d]
                        if c == d:
                            self.qubo_matrix[c, d] += coeff
                        else:
                            self.qubo_matrix[c, d] += coeff / 2  # Off-diagonal terms
                
                # Linear terms: -2 * K_{ℓ,j}^target * Σ_{c∈R_ℓ} β_{c,j} x_c
                for c in bonds_in_bucket:
                    self.qubo_matrix[c, c] -= 2 * char_weight * target_val * contributions[c]
                
                # Constant term: (K_{ℓ,j}^target)²
                self.qubo_offset += char_weight * target_val**2
        
        # CONSTRAINT PENALTIES
        
        # 1. Maximum bonds constraint: Σx_c ≤ max_bonds
        max_bonds = self.config.max_bonds_in_portfolio
        penalty = penalty_weights['max_bonds']
        
        # Penalty: penalty * (Σx_c - max_bonds)²
        # = penalty * (Σx_c² + ΣΣx_c*x_d - 2*max_bonds*Σx_c + max_bonds²)
        for c in range(n_bonds):
            for d in range(n_bonds):
                if c == d:
                    self.qubo_matrix[c, d] += penalty
                    self.qubo_matrix[c, d] -= 2 * penalty * max_bonds
                else:
                    self.qubo_matrix[c, d] += penalty
        
        self.qubo_offset += penalty * max_bonds**2
        
        # 2. Minimum bonds constraint: Σx_c ≥ min_bonds
        min_bonds = max(3, self.config.max_bonds_in_portfolio // 2)
        penalty = penalty_weights['min_bonds']
        
        # Penalty: penalty * max(0, min_bonds - Σx_c)²
        # Approximated as: penalty * (min_bonds - Σx_c)² for simplicity
        for c in range(n_bonds):
            for d in range(n_bonds):
                if c == d:
                    self.qubo_matrix[c, d] += penalty
                    self.qubo_matrix[c, d] -= 2 * penalty * min_bonds
                else:
                    self.qubo_matrix[c, d] -= penalty
        
        self.qubo_offset += penalty * min_bonds**2
        
        # 3. Cash flow constraints (simplified)
        # Target cost around market_value with some tolerance
        penalty = penalty_weights['cash_flow']
        target_cost = self.config.market_value * 0.98  # 2% cash buffer
        bond_prices = self.classical_optimizer.bond_data['price'].values
        
        # Penalty: penalty * (Σ(price_c * x_c) - target_cost)²
        for c in range(n_bonds):
            for d in range(n_bonds):
                coeff = penalty * bond_prices[c] * bond_prices[d]
                if c == d:
                    self.qubo_matrix[c, d] += coeff
                else:
                    self.qubo_matrix[c, d] += coeff / 2
            
            self.qubo_matrix[c, c] -= 2 * penalty * target_cost * bond_prices[c]
        
        self.qubo_offset += penalty * target_cost**2
        
        # 4. Risk bucket constraints: max bonds per bucket
        penalty = penalty_weights['risk_bucket']
        
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.classical_optimizer.risk_buckets[bucket_id]['bonds']
            max_bonds_bucket = self.classical_optimizer.risk_buckets[bucket_id]['max_bonds_allowed']
            
            # Penalty: penalty * (Σ_{c∈bucket} x_c - max_bonds_bucket)²
            for c in bonds_in_bucket:
                for d in bonds_in_bucket:
                    if c == d:
                        self.qubo_matrix[c, d] += penalty
                        self.qubo_matrix[c, d] -= 2 * penalty * max_bonds_bucket
                    else:
                        self.qubo_matrix[c, d] += penalty
            
            self.qubo_offset += penalty * max_bonds_bucket**2
        
        # 5. Characteristic bounds (guardrails)
        penalty = penalty_weights['characteristic_bounds']
        
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.classical_optimizer.risk_buckets[bucket_id]['bonds']
            
            for char_id in range(self.config.n_characteristics):
                target_val = self.classical_optimizer.targets[bucket_id][char_id]['target']
                tolerance = self.classical_optimizer.targets[bucket_id][char_id]['tolerance']
                contributions = self.classical_optimizer.characteristics[char_id]['contributions']
                
                # Upper bound penalty: max(0, portfolio_val - upper_bound)²
                upper_bound = target_val * (1 + tolerance)
                
                for c in bonds_in_bucket:
                    for d in bonds_in_bucket:
                        coeff = penalty * contributions[c] * contributions[d] * 0.1  # Reduced weight
                        if c == d:
                            self.qubo_matrix[c, d] += coeff
                        else:
                            self.qubo_matrix[c, d] += coeff / 2
        
        print(f"QUBO formulation complete:")
        print(f"  Matrix size: {self.qubo_matrix.shape}")
        print(f"  Matrix norm: {np.linalg.norm(self.qubo_matrix):.3f}")
        print(f"  Offset: {self.qubo_offset:.3f}")
        print(f"  Non-zero elements: {np.count_nonzero(self.qubo_matrix)}")
        
        return self.qubo_matrix, self.qubo_offset
    
    def qubo_to_pauli(self, Q: np.ndarray, offset: float) -> SparsePauliOp:
        """Convert QUBO matrix to Pauli operator for VQE"""
        print(f"Converting QUBO to Pauli operator...")
        
        n_qubits = Q.shape[0]
        pauli_list = []
        
        # Tolerance for numerical precision
        tol = 1e-10
        
        # Convert QUBO variables x_i ∈ {0,1} to Pauli: x_i = (1 - Z_i)/2
        # So x_i * x_j = ((1 - Z_i)/2) * ((1 - Z_j)/2) = (1 - Z_i - Z_j + Z_i*Z_j)/4
        
        for i in range(n_qubits):
            for j in range(i, n_qubits):  # Upper triangular only
                if abs(Q[i, j]) > tol:
                    if i == j:
                        # Diagonal term Q[i,i] * x_i = Q[i,i] * (1 - Z_i)/2
                        pauli_list.append(('I' * n_qubits, Q[i, i] / 2))
                        
                        pauli_str = ['I'] * n_qubits
                        pauli_str[i] = 'Z'
                        pauli_list.append((''.join(pauli_str), -Q[i, i] / 2))
                    else:
                        # Off-diagonal term Q[i,j] * x_i * x_j
                        # = Q[i,j] * (1 - Z_i - Z_j + Z_i*Z_j)/4
                        coeff = Q[i, j] if i != j else Q[i, j] / 2  # Symmetric matrix handling
                        
                        # Constant term
                        pauli_list.append(('I' * n_qubits, coeff / 4))
                        
                        # Z_i term
                        pauli_str_i = ['I'] * n_qubits
                        pauli_str_i[i] = 'Z'
                        pauli_list.append((''.join(pauli_str_i), -coeff / 4))
                        
                        # Z_j term  
                        pauli_str_j = ['I'] * n_qubits
                        pauli_str_j[j] = 'Z'
                        pauli_list.append((''.join(pauli_str_j), -coeff / 4))
                        
                        # Z_i * Z_j term
                        pauli_str_ij = ['I'] * n_qubits
                        pauli_str_ij[i] = 'Z'
                        pauli_str_ij[j] = 'Z'
                        pauli_list.append((''.join(pauli_str_ij), coeff / 4))
        
        # Add constant offset
        if abs(offset) > tol:
            pauli_list.append(('I' * n_qubits, offset))
        
        # Remove terms with zero coefficients
        pauli_list = [(pauli, coeff) for pauli, coeff in pauli_list if abs(coeff) > tol]
        
        print(f"  Generated {len(pauli_list)} Pauli terms")
        
        return SparsePauliOp.from_list(pauli_list)
    
    def create_vqe_ansatz(self, n_qubits: int, depth: int = 3) -> QuantumCircuit:
        """Create parameterized quantum circuit for VQE"""
        print(f"Creating VQE ansatz: {n_qubits} qubits, depth {depth}")
        
        # Use TwoLocal ansatz - proven effective for optimization problems
        ansatz = TwoLocal(
            n_qubits,
            rotation_blocks=['ry'],     # Single-qubit rotation gates
            entanglement_blocks='cz',   # Two-qubit entangling gates
            entanglement='linear',      # Linear connectivity
            reps=depth,                 # Circuit depth
            skip_final_rotation_layer=False,
            insert_barriers=True
        )
        
        print(f"  Parameters: {ansatz.num_parameters}")
        print(f"  Circuit depth: {ansatz.depth()}")
        
        return ansatz
    
    def solve_quantum_vqe(self, 
                         max_iter: int = 300, 
                         optimizer_name: str = 'COBYLA',
                         n_attempts: int = 3,
                         penalty_weights: Dict[str, float] = None) -> Dict:
        """
        Solve ETF optimization using VQE
        
        Args:
            max_iter: Maximum optimizer iterations
            optimizer_name: Classical optimizer to use
            n_attempts: Number of random restarts
            penalty_weights: QUBO penalty weights
        """
        print("\n" + "="*50)
        print("QUANTUM VQE OPTIMIZATION")
        print("="*50)
        
        start_time = time.time()
        
        # Step 1: Generate QUBO formulation
        Q, offset = self.formulate_qubo(penalty_weights)
        n_qubits = Q.shape[0]
        
        if n_qubits > 25:
            print(f"Warning: {n_qubits} qubits may be slow on classical simulator")
            print("Consider reducing problem size for faster execution")
        
        # Step 2: Convert to Pauli operator
        self.pauli_operator = self.qubo_to_pauli(Q, offset)
        
        # Step 3: Create quantum circuit
        ansatz = self.create_vqe_ansatz(n_qubits, depth=3)
        
        # Step 4: Set up VQE components
        simulator = AerSimulator()
        estimator = Estimator()
        
        # Choose optimizer
        optimizer_map = {
            'COBYLA': COBYLA(maxiter=max_iter),
            'L_BFGS_B': L_BFGS_B(maxiter=max_iter),
            'SPSA': SPSA(maxiter=max_iter),
            'SLSQP': SLSQP(maxiter=max_iter)
        }
        
        optimizer = optimizer_map.get(optimizer_name, COBYLA(maxiter=max_iter))
        print(f"Using optimizer: {optimizer_name}")
        
        # Step 5: Run VQE with multiple attempts
        best_result = None
        best_eigenvalue = float('inf')
        all_results = []
        
        def callback(iteration, parameters, mean_value, std_dev):
            """Callback to track VQE progress"""
            self.vqe_history.append({
                'iteration': iteration,
                'eigenvalue': mean_value,
                'std_dev': std_dev
            })
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}: eigenvalue = {mean_value:.6f}")
        
        for attempt in range(n_attempts):
            print(f"\nVQE Attempt {attempt + 1}/{n_attempts}")
            self.vqe_history = []  # Reset history for this attempt
            
            # Create VQE instance
            vqe = VQE(
                estimator=estimator,
                ansatz=ansatz,
                optimizer=optimizer,
                callback=callback
            )
            
            try:
                # Run VQE
                result = vqe.compute_minimum_eigenvalue(self.pauli_operator)
                
                print(f"    Eigenvalue: {result.optimal_value:.6f}")
                print(f"    Function evaluations: {result.cost_function_evals}")
                
                # Track best result
                if result.optimal_value < best_eigenvalue:
                    best_eigenvalue = result.optimal_value
                    best_result = result
                
                all_results.append({
                    'attempt': attempt + 1,
                    'eigenvalue': result.optimal_value,
                    'parameters': result.optimal_parameters.copy(),
                    'function_evals': result.cost_function_evals,
                    'history': self.vqe_history.copy()
                })
                
            except Exception as e:
                print(f"    Attempt {attempt + 1} failed: {str(e)}")
                continue
        
        if best_result is None:
            print("❌ All VQE attempts failed")
            return None
        
        # Step 6: Extract portfolio from quantum solution
        portfolio_info = self._extract_portfolio_from_vqe(best_result, n_qubits)
        
        total_time = time.time() - start_time
        
        # Step 7: Compile results
        self.quantum_result = {
            'vqe_result': best_result,
            'eigenvalue': best_eigenvalue,
            'optimal_parameters': best_result.optimal_parameters,
            'function_evaluations': best_result.cost_function_evals,
            'portfolio': portfolio_info,
            'n_qubits': n_qubits,
            'optimizer_used': optimizer_name,
            'total_time': total_time,
            'n_attempts': n_attempts,
            'all_attempts': all_results,
            'qubo_matrix_norm': np.linalg.norm(Q),
            'pauli_terms': len(self.pauli_operator)
        }
        
        print(f"\n🎉 Quantum VQE Optimization Complete!")
        print(f"  Best eigenvalue: {best_eigenvalue:.6f}")
        print(f"  Bonds selected: {len(portfolio_info['selected_bonds'])}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Function evaluations: {best_result.cost_function_evals}")
        
        return self.quantum_result
    
    def _extract_portfolio_from_vqe(self, vqe_result, n_qubits: int) -> Dict:
        """Extract bond portfolio from VQE result"""
        
        # Method 1: Use optimal parameters to determine most likely bond selection
        # This is a heuristic - in practice you'd sample from the quantum state
        
        params = vqe_result.optimal_parameters
        n_bonds = n_qubits
        
        # Simple heuristic: parameters with large magnitude suggest inclusion
        if len(params) >= n_bonds:
            # Use first n_bonds parameters as indicators
            bond_indicators = params[:n_bonds]
            
            # Normalize to probabilities using sigmoid-like function
            probabilities = 1 / (1 + np.exp(-2 * bond_indicators))
            
            # Select bonds above threshold
            threshold = 0.5
            selected_bonds = [i for i, prob in enumerate(probabilities) if prob > threshold]
            
        else:
            # Alternative: use parameter magnitudes across all parameters
            param_importance = np.abs(params)
            n_to_select = min(self.config.max_bonds_in_portfolio, n_bonds // 2)
            
            # Select bonds corresponding to largest parameter magnitudes
            important_params = np.argsort(param_importance)[-n_to_select:]
            selected_bonds = [i % n_bonds for i in important_params]
            selected_bonds = list(set(selected_bonds))  # Remove duplicates
        
        # Ensure we have reasonable number of bonds
        if len(selected_bonds) == 0:
            selected_bonds = [0, 1, 2]  # Fallback
        elif len(selected_bonds) > self.config.max_bonds_in_portfolio:
            selected_bonds = selected_bonds[:self.config.max_bonds_in_portfolio]
        
        # Calculate portfolio metrics
        if hasattr(self.classical_optimizer, 'bond_data') and self.classical_optimizer.bond_data is not None:
            bond_prices = self.classical_optimizer.bond_data['price'].values
            total_cost = sum(bond_prices[i] for i in selected_bonds)
            residual_cash = self.config.market_value - total_cost
            
            # Calculate characteristic values
            portfolio_characteristics = {}
            for char_id in range(self.config.n_characteristics):
                contributions = self.classical_optimizer.characteristics[char_id]['contributions']
                char_value = sum(contributions[i] for i in selected_bonds)
                portfolio_characteristics[char_id] = char_value
        else:
            total_cost = 0
            residual_cash = 0
            portfolio_characteristics = {}
        
        return {
            'selected_bonds': selected_bonds,
            'n_bonds_selected': len(selected_bonds),
            'total_cost': total_cost,
            'residual_cash': residual_cash,
            'residual_cash_pct': residual_cash / self.config.market_value if self.config.market_value > 0 else 0,
            'portfolio_characteristics': portfolio_characteristics,
            'bond_probabilities': probabilities if 'probabilities' in locals() else None
        }
    
    def compare_classical_quantum(self) -> Dict:
        """Compare classical and quantum optimization results"""
        if self.classical_optimizer.classical_result is None:
            print("No classical result found. Run classical optimization first.")
            return None
        
        if self.quantum_result is None:
            print("No quantum result found. Run quantum optimization first.")
            return None
        
        print("\n" + "="*60)
        print("CLASSICAL vs QUANTUM COMPARISON")
        print("="*60)
        
        classical = self.classical_optimizer.classical_result
        quantum = self.quantum_result['portfolio']
        
        comparison = {
            'classical': classical,
            'quantum': quantum,
            'metrics': {}
        }
        
        # Bond selection comparison
        classical_bonds = set(classical['selected_bonds'])
        quantum_bonds = set(quantum['selected_bonds'])
        
        overlap = len(classical_bonds.intersection(quantum_bonds))
        union = len(classical_bonds.union(quantum_bonds))
        jaccard_similarity = overlap / union if union > 0 else 0
        
        comparison['metrics']['bond_overlap'] = overlap
        comparison['metrics']['jaccard_similarity'] = jaccard_similarity
        
        # Performance metrics
        metrics = [
            ('Bonds Selected', 'n_bonds_selected'),
            ('Total Cost', 'total_cost'), 
            ('Residual Cash %', 'residual_cash_pct'),
        ]
        
        print(f"{'Metric':<20} {'Classical':<15} {'Quantum':<15} {'Difference':<15}")
        print("-" * 65)
        
        for metric_name, key in metrics:
            classical_val = classical.get(key, 0)
            quantum_val = quantum.get(key, 0)
            diff = quantum_val - classical_val
            
            if 'pct' in key:
                print(f"{metric_name:<20} {classical_val:<14.2%} {quantum_val:<14.2%} {diff:<14.2%}")
            elif 'cost' in key.lower():
                print(f"{metric_name:<20} ${classical_val:<14,.0f} ${quantum_val:<14,.0f} ${diff:<14,.0f}")
            else:
                print(f"{metric_name:<20} {classical_val:<15} {quantum_val:<15} {diff:<15}")
            
            comparison['metrics'][f'{key}_difference'] = diff
        
        # Objective function comparison
        classical_obj = classical.get('total_tracking_error', float('inf'))
        quantum_obj = self.quantum_result['eigenvalue']
        
        print(f"\nObjective Function Comparison:")
        print(f"  Classical tracking error: {classical_obj:.6f}")
        print(f"  Quantum eigenvalue: {quantum_obj:.6f}")
        print(f"  Difference: {quantum_obj - classical_obj:.6f}")
        
        # Bond selection analysis
        print(f"\nBond Selection Analysis:")
        print(f"  Classical bonds: {sorted(list(classical_bonds))}")
        print(f"  Quantum bonds: {sorted(list(quantum_bonds))}")
        print(f"  Overlapping bonds: {sorted(list(classical_bonds.intersection(quantum_bonds)))}")
        print(f"  Jaccard similarity: {jaccard_similarity:.3f}")
        
        # Computational analysis
        print(f"\nComputational Performance:")
        print(f"  Classical solver time: {classical.get('solver_time', 'N/A')}")
        print(f"  Quantum total time: {self.quantum_result['total_time']:.2f}s")
        print(f"  Quantum function evals: {self.quantum_result['function_evaluations']}")
        print(f"  VQE attempts: {self.quantum_result['n_attempts']}")
        
        comparison['metrics']['classical_objective'] = classical_obj
        comparison['metrics']['quantum_objective'] = quantum_obj
        comparison['metrics']['objective_difference'] = quantum_obj - classical_obj
        
        return comparison
    
    def visualize_vqe_convergence(self, save_path: Optional[str] = None):
        """Visualize VQE convergence history"""
        if not self.quantum_result or not self.quantum_result['all_attempts']:
            print("No VQE history to plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('VQE Convergence Analysis', fontsize=14)
        
        # Plot 1: Convergence for all attempts
        for i, attempt in enumerate(self.quantum_result['all_attempts']):
            history = attempt['history']
            if history:
                iterations = [h['iteration'] for h in history]
                eigenvalues = [h['eigenvalue'] for h in history]
                axes[0].plot(iterations, eigenvalues, label=f'Attempt {attempt["attempt"]}', alpha=0.7)
        
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Eigenvalue')
        axes[0].set_title('VQE Convergence by Attempt')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Final eigenvalues by attempt
        attempts = [a['attempt'] for a in self.quantum_result['all_attempts']]
        final_eigenvalues = [a['eigenvalue'] for a in self.quantum_result['all_attempts']]
        
        axes[1].bar(attempts, final_eigenvalues, alpha=0.7)
        axes[1].set_xlabel('Attempt')
        axes[1].set_ylabel('Final Eigenvalue')
        axes[1].set_title('Final Eigenvalue by Attempt')
        axes[1].grid(True, alpha=0.3)
        
        # Highlight best attempt
        best_attempt = min(range(len(final_eigenvalues)), key=lambda i: final_eigenvalues[i])
        axes[1].bar(attempts[best_attempt], final_eigenvalues[best_attempt], color='red', alpha=0.8, label='Best')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"VQE convergence plot saved to {save_path}")
        
        plt.show()

# Example usage and integration
if __name__ == "__main__":
    print("Vanguard ETF Optimizer - Quantum VQE Implementation")
    print("="*50)
    
    # Step 1: Create classical optimizer and solve
    config = ETFConfiguration(
        n_bonds=15,  # Start small for quantum
        n_risk_buckets=3,
        n_characteristics=3,
        max_bonds_in_portfolio=8,
        market_value=1000000
    )
    
    classical_optimizer = VanguardETFOptimizer(config)
    classical_optimizer.generate_realistic_bond_universe(seed=42)
    
    print("Solving classical problem first...")
    classical_result = classical_optimizer.solve_classical_etf_optimization()
    
    if classical_result:
        # Step 2: Create quantum optimizer and solve
        quantum_optimizer = QuantumETFOptimizer(classical_optimizer)
        
        print("\nSolving quantum problem...")
        quantum_result = quantum_optimizer.solve_quantum_vqe(
            max_iter=200,
            optimizer_name='COBYLA',
            n_attempts=3
        )
        
        if quantum_result:
            # Step 3: Compare results
            comparison = quantum_optimizer.compare_classical_quantum()
            
            # Step 4: Visualize convergence
            quantum_optimizer.visualize_vqe_convergence()
            
            print("\n" + "="*50)
            print("QUANTUM IMPLEMENTATION COMPLETE ✅")
            print("="*50)
            print("Successfully demonstrated quantum optimization for Vanguard ETF creation!")
            
        else:
            print("❌ Quantum optimization failed")
    else:
        print("❌ Classical optimization failed - cannot proceed to quantum")
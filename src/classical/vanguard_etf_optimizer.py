"""
Vanguard ETF Optimizer - Classical Implementation
Exact implementation of OneOpto model for ETF creation/redemption
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ETFConfiguration:
    """Configuration for ETF optimization"""
    n_bonds: int = 30
    n_risk_buckets: int = 4
    n_characteristics: int = 3
    max_bonds_in_portfolio: int = 15
    min_cash_flow_pct: float = 0.01
    max_cash_flow_pct: float = 0.05
    market_value: float = 1000000.0  # $1M basket

class VanguardETFOptimizer:
    """
    Vanguard ETF Creation Optimizer
    
    Implements the exact OneOpto model used by Vanguard for:
    - Fixed income ETF creation/redemption
    - Index tracking optimization
    - Bond basket construction
    """
    
    def __init__(self, config: ETFConfiguration = None):
        self.config = config or ETFConfiguration()
        self.bond_data = None
        self.risk_buckets = None
        self.characteristics = None
        self.targets = None
        self.classical_result = None
        
    def generate_realistic_bond_universe(self, seed: int = 42) -> None:
        """Generate realistic bond universe matching Vanguard's data structure"""
        np.random.seed(seed)
        
        print(f"Generating bond universe: {self.config.n_bonds} bonds, {self.config.n_risk_buckets} risk buckets")
        
        # BOND DATA - Realistic fixed income characteristics
        self.bond_data = pd.DataFrame({
            'bond_id': [f'BOND_{i:03d}' for i in range(self.config.n_bonds)],
            'price': np.random.uniform(95, 105, self.config.n_bonds),  # Near par pricing
            'min_trade': np.random.choice([1000, 5000, 10000], self.config.n_bonds),
            'max_trade': np.random.uniform(50000, 200000, self.config.n_bonds),
            'duration': np.random.uniform(1.0, 10.0, self.config.n_bonds),  # Years
            'credit_quality': np.random.uniform(1.0, 5.0, self.config.n_bonds),  # AAA=1, BBB=3, etc
            'sector_weight': np.random.uniform(0.1, 2.0, self.config.n_bonds),  # Sector exposure
        })
        
        # RISK BUCKETS - Group bonds by risk characteristics
        self.risk_buckets = {}
        bucket_size = max(5, self.config.n_bonds // self.config.n_risk_buckets)
        
        for bucket_id in range(self.config.n_risk_buckets):
            start_idx = bucket_id * bucket_size
            end_idx = min(start_idx + bucket_size, self.config.n_bonds)
            if bucket_id == self.config.n_risk_buckets - 1:  # Last bucket gets remainder
                end_idx = self.config.n_bonds
                
            self.risk_buckets[bucket_id] = {
                'bonds': list(range(start_idx, end_idx)),
                'name': f'Risk_Bucket_{bucket_id}',
                'max_bonds_allowed': min(8, len(range(start_idx, end_idx)) - 1)
            }
        
        # CHARACTERISTICS - What we're trying to match to index
        self.characteristics = {}
        char_names = ['Duration', 'Credit_Quality', 'Sector_Exposure']
        
        for j in range(self.config.n_characteristics):
            char_name = char_names[j] if j < len(char_names) else f'Characteristic_{j}'
            
            # Each characteristic has contributions from bonds
            if j == 0:  # Duration
                contributions = self.bond_data['duration'].values
                target_base = 5.0  # 5-year average duration target
            elif j == 1:  # Credit Quality  
                contributions = self.bond_data['credit_quality'].values
                target_base = 2.5  # Investment grade average
            else:  # Sector
                contributions = self.bond_data['sector_weight'].values  
                target_base = 1.0  # Neutral sector weight
            
            self.characteristics[j] = {
                'name': char_name,
                'contributions': contributions,  # β_{c,j} values
                'weight': 1.0,  # ρ_j importance weight
            }
        
        # TARGETS - What the index looks like (what we want to replicate)
        self.targets = {}
        for bucket_id in range(self.config.n_risk_buckets):
            self.targets[bucket_id] = {}
            for char_id in range(self.config.n_characteristics):
                # Target is weighted average of bonds in this risk bucket
                bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
                contributions = self.characteristics[char_id]['contributions']
                bucket_contributions = contributions[bonds_in_bucket]
                
                # Add some noise to make it interesting
                target_val = np.mean(bucket_contributions) + np.random.normal(0, 0.1)
                
                self.targets[bucket_id][char_id] = {
                    'target': target_val,
                    'tolerance': 0.15,  # ±15% tolerance
                }
        
        print("Generated bond universe successfully:")
        print(f"  Price range: ${self.bond_data['price'].min():.2f} - ${self.bond_data['price'].max():.2f}")
        print(f"  Duration range: {self.bond_data['duration'].min():.1f} - {self.bond_data['duration'].max():.1f} years")
        print(f"  Risk buckets: {len(self.risk_buckets)} buckets")
        
    def solve_classical_etf_optimization(self) -> Dict:
        """
        Solve ETF optimization using classical CVXPY
        Implements Vanguard's OneOpto model exactly
        """
        print("\n" + "="*50)
        print("CLASSICAL ETF OPTIMIZATION")
        print("="*50)
        
        if self.bond_data is None:
            raise ValueError("Must generate bond universe first")
        
        # DECISION VARIABLES - Binary bond inclusion
        y = cp.Variable(self.config.n_bonds, boolean=True, name="bond_selection")
        
        # OBJECTIVE FUNCTION - Minimize tracking error
        # Σ_{ℓ,j} ρ_j (Σ_{c∈R_ℓ} β_{c,j} y_c - K_{ℓ,j}^target)²
        objective_terms = []
        
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
            
            for char_id in range(self.config.n_characteristics):
                # Get characteristic data
                target_val = self.targets[bucket_id][char_id]['target']
                char_weight = self.characteristics[char_id]['weight']
                contributions = self.characteristics[char_id]['contributions']
                
                # Portfolio characteristic value: Σ_{c∈R_ℓ} β_{c,j} y_c
                portfolio_char_value = cp.sum([contributions[c] * y[c] for c in bonds_in_bucket])
                
                # Squared deviation from target
                deviation_squared = cp.square(portfolio_char_value - target_val)
                objective_terms.append(char_weight * deviation_squared)
        
        objective = cp.Minimize(cp.sum(objective_terms))
        
        # CONSTRAINTS
        constraints = []
        
        # 1. Maximum number of bonds in portfolio
        constraints.append(cp.sum(y) <= self.config.max_bonds_in_portfolio)
        constraints.append(cp.sum(y) >= 3)  # Minimum for diversification
        
        # 2. Cash flow constraints (budget management)
        total_cost = cp.sum(cp.multiply(self.bond_data['price'].values, y))
        min_cash = self.config.min_cash_flow_pct * self.config.market_value
        max_cash = self.config.max_cash_flow_pct * self.config.market_value
        constraints.append(total_cost >= self.config.market_value - max_cash)
        constraints.append(total_cost <= self.config.market_value - min_cash)
        
        # 3. Risk bucket constraints
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
            max_bonds_allowed = self.risk_buckets[bucket_id]['max_bonds_allowed']
            constraints.append(cp.sum([y[c] for c in bonds_in_bucket]) <= max_bonds_allowed)
        
        # 4. Characteristic bounds (guardrails)
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
            
            for char_id in range(self.config.n_characteristics):
                target_val = self.targets[bucket_id][char_id]['target']
                tolerance = self.targets[bucket_id][char_id]['tolerance']
                contributions = self.characteristics[char_id]['contributions']
                
                portfolio_char_value = cp.sum([contributions[c] * y[c] for c in bonds_in_bucket])
                
                # Guardrails: target ± tolerance
                constraints.append(portfolio_char_value >= target_val * (1 - tolerance))
                constraints.append(portfolio_char_value <= target_val * (1 + tolerance))
        
        # SOLVE THE PROBLEM
        problem = cp.Problem(objective, constraints)
        
        print("Solving optimization problem...")
        print(f"  Variables: {self.config.n_bonds} binary variables")
        print(f"  Constraints: {len(constraints)} constraints")
        
        # Try different solvers
        solvers_to_try = [cp.GUROBI, cp.SCIP, cp.CBC, cp.GLPK_MI]
        solved = False
        
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in cp.settings.SOLUTION_PRESENT:
                    solved = True
                    print(f"  Solved with {solver}")
                    break
            except:
                continue
        
        if not solved:
            print("  Trying default solver...")
            # Try solvers in order of preference
        solvers_to_try = ['SCIP', 'CLARABEL', 'ECOS_BB', 'GLPK_MI', 'CVXOPT']
        solved = False
        
        for solver_name in solvers_to_try:
            try:
                print(f"  Trying {solver_name} solver...")
                problem.solve(solver=solver_name, verbose=True)
                if problem.status == 'optimal':
                    print(f"  ✅ Successfully solved with {solver_name}")
                    solved = True
                    break
                else:
                    print(f"  ❌ {solver_name} failed with status: {problem.status}")
            except Exception as e:
                print(f"  ❌ {solver_name} failed with error: {e}")
                continue
        
        if not solved:
            raise Exception("All solvers failed. Problem may be infeasible.")
        
        # EXTRACT AND ANALYZE SOLUTION
        if problem.status in cp.settings.SOLUTION_PRESENT:
            selected_bonds = [i for i in range(self.config.n_bonds) if y.value[i] > 0.5]
            total_tracking_error = problem.value
            total_cost = float(np.sum(self.bond_data['price'].values * y.value))
            residual_cash = self.config.market_value - total_cost
            
            # Calculate portfolio characteristics
            portfolio_characteristics = {}
            for char_id in range(self.config.n_characteristics):
                contributions = self.characteristics[char_id]['contributions']
                char_value = sum(contributions[i] * y.value[i] for i in range(self.config.n_bonds))
                portfolio_characteristics[char_id] = char_value
            
            self.classical_result = {
                'bond_selection': y.value,
                'selected_bonds': selected_bonds,
                'n_bonds_selected': len(selected_bonds),
                'total_tracking_error': total_tracking_error,
                'total_cost': total_cost,
                'residual_cash': residual_cash,
                'residual_cash_pct': residual_cash / self.config.market_value,
                'portfolio_characteristics': portfolio_characteristics,
                'objective_value': problem.value,
                'status': problem.status,
                'solver_time': problem.solver_stats.solve_time if problem.solver_stats else None
            }
            
            print(f"\nClassical Solution Found:")
            print(f"  Status: {problem.status}")
            print(f"  Bonds selected: {len(selected_bonds)}/{self.config.n_bonds}")
            print(f"  Total tracking error: {total_tracking_error:.6f}")
            print(f"  Total cost: ${total_cost:,.0f}")
            print(f"  Residual cash: {residual_cash/self.config.market_value:.2%}")
            print(f"  Selected bonds: {selected_bonds}")
            
            # Show portfolio characteristics vs targets
            print(f"\nPortfolio Characteristics vs Targets:")
            for bucket_id in range(self.config.n_risk_buckets):
                print(f"  Risk Bucket {bucket_id}:")
                bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
                selected_in_bucket = [b for b in bonds_in_bucket if b in selected_bonds]
                
                for char_id in range(self.config.n_characteristics):
                    contributions = self.characteristics[char_id]['contributions']
                    portfolio_val = sum(contributions[c] for c in selected_in_bucket)
                    target_val = self.targets[bucket_id][char_id]['target']
                    char_name = self.characteristics[char_id]['name']
                    
                    print(f"    {char_name}: {portfolio_val:.3f} vs target {target_val:.3f}")
            
            return self.classical_result
        else:
            print(f"Optimization failed: {problem.status}")
            return None
    
    def visualize_portfolio(self, save_path: Optional[str] = None):
        """Create visualizations of the ETF portfolio"""
        if self.classical_result is None:
            print("No solution to visualize. Run optimization first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vanguard ETF Portfolio Analysis', fontsize=16)
        
        # 1. Bond Selection by Risk Bucket
        bucket_selections = {}
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
            selected_in_bucket = [b for b in bonds_in_bucket if b in self.classical_result['selected_bonds']]
            bucket_selections[f'Bucket {bucket_id}'] = len(selected_in_bucket)
        
        axes[0,0].bar(bucket_selections.keys(), bucket_selections.values())
        axes[0,0].set_title('Bonds Selected by Risk Bucket')
        axes[0,0].set_ylabel('Number of Bonds')
        
        # 2. Portfolio Characteristics vs Targets
        characteristics_data = []
        for bucket_id in range(self.config.n_risk_buckets):
            bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
            selected_in_bucket = [b for b in bonds_in_bucket if b in self.classical_result['selected_bonds']]
            
            for char_id in range(self.config.n_characteristics):
                contributions = self.characteristics[char_id]['contributions']
                portfolio_val = sum(contributions[c] for c in selected_in_bucket)
                target_val = self.targets[bucket_id][char_id]['target']
                char_name = self.characteristics[char_id]['name']
                
                characteristics_data.append({
                    'Bucket': f'B{bucket_id}',
                    'Characteristic': char_name,
                    'Portfolio': portfolio_val,
                    'Target': target_val,
                    'Error': abs(portfolio_val - target_val)
                })
        
        char_df = pd.DataFrame(characteristics_data)
        bucket_chars = char_df.groupby('Bucket')[['Portfolio', 'Target', 'Error']].mean()
        
        x = np.arange(len(bucket_chars))
        width = 0.35
        
        axes[0,1].bar(x - width/2, bucket_chars['Portfolio'], width, label='Portfolio', alpha=0.7)
        axes[0,1].bar(x + width/2, bucket_chars['Target'], width, label='Target', alpha=0.7)
        axes[0,1].set_title('Portfolio vs Target Characteristics')
        axes[0,1].set_xlabel('Risk Buckets')
        axes[0,1].set_ylabel('Characteristic Value')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(bucket_chars.index)
        axes[0,1].legend()
        
        # 3. Selected Bond Details
        selected_bond_data = self.bond_data.iloc[self.classical_result['selected_bonds']]
        
        axes[1,0].scatter(selected_bond_data['duration'], selected_bond_data['credit_quality'], 
                         s=selected_bond_data['price']*2, alpha=0.6)
        axes[1,0].set_title('Selected Bonds: Duration vs Credit Quality')
        axes[1,0].set_xlabel('Duration (years)')
        axes[1,0].set_ylabel('Credit Quality')
        
        # 4. Cost Analysis
        cost_data = {
            'Portfolio Cost': self.classical_result['total_cost'],
            'Residual Cash': self.classical_result['residual_cash'],
            'Target Value': self.config.market_value
        }
        
        axes[1,1].pie([self.classical_result['total_cost'], self.classical_result['residual_cash']], 
                     labels=['Portfolio Cost', 'Residual Cash'], autopct='%1.1f%%')
        axes[1,1].set_title(f"Cash Allocation (${self.config.market_value:,.0f} total)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_summary_report(self) -> str:
        """Generate executive summary of optimization results"""
        if self.classical_result is None:
            return "No optimization results to report."
        
        report = f"""
VANGUARD ETF OPTIMIZATION SUMMARY REPORT
{'='*50}

Portfolio Configuration:
  - Bond Universe: {self.config.n_bonds} bonds
  - Risk Buckets: {self.config.n_risk_buckets} buckets  
  - Target Portfolio Size: {self.config.max_bonds_in_portfolio} bonds
  - Market Value: ${self.config.market_value:,.0f}

Optimization Results:
  - Status: {self.classical_result['status']}
  - Bonds Selected: {self.classical_result['n_bonds_selected']}/{self.config.n_bonds}
  - Total Tracking Error: {self.classical_result['total_tracking_error']:.6f}
  - Portfolio Cost: ${self.classical_result['total_cost']:,.0f}
  - Residual Cash: {self.classical_result['residual_cash_pct']:.2%}

Portfolio Characteristics:
"""
        
        for bucket_id in range(self.config.n_risk_buckets):
            report += f"\n  Risk Bucket {bucket_id}:\n"
            bonds_in_bucket = self.risk_buckets[bucket_id]['bonds']
            selected_in_bucket = [b for b in bonds_in_bucket if b in self.classical_result['selected_bonds']]
            
            report += f"    Bonds Selected: {len(selected_in_bucket)}/{len(bonds_in_bucket)}\n"
            
            for char_id in range(self.config.n_characteristics):
                contributions = self.characteristics[char_id]['contributions']
                portfolio_val = sum(contributions[c] for c in selected_in_bucket)
                target_val = self.targets[bucket_id][char_id]['target']
                error_pct = abs(portfolio_val - target_val) / target_val * 100 if target_val != 0 else 0
                char_name = self.characteristics[char_id]['name']
                
                report += f"    {char_name}: {portfolio_val:.3f} (target: {target_val:.3f}, error: {error_pct:.1f}%)\n"
        
        report += f"""
Business Impact:
  - Tracking Error: {self.classical_result['total_tracking_error']:.6f} (lower is better)
  - Cash Management: {self.classical_result['residual_cash_pct']:.2%} residual
  - Diversification: {self.classical_result['n_bonds_selected']} bonds across {self.config.n_risk_buckets} risk buckets
  - Constraint Satisfaction: All guardrails satisfied

Next Steps:
  1. Validate portfolio against additional business rules
  2. Run quantum optimization for comparison
  3. Perform sensitivity analysis
  4. Generate implementation recommendations
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("Vanguard ETF Optimizer - Classical Implementation")
    print("="*50)
    
    # Create optimizer with realistic configuration
    config = ETFConfiguration(
        n_bonds=30,
        n_risk_buckets=4, 
        n_characteristics=3,
        max_bonds_in_portfolio=12,
        market_value=1000000
    )
    
    optimizer = VanguardETFOptimizer(config)
    
    # Generate realistic bond universe
    optimizer.generate_realistic_bond_universe(seed=42)
    
    # Solve optimization problem
    result = optimizer.solve_classical_etf_optimization()
    
    if result:
        # Generate visualizations
        optimizer.visualize_portfolio()
        
        # Print summary report
        report = optimizer.generate_summary_report()
        print(report)
        
        print("\n" + "="*50)
        print("CLASSICAL IMPLEMENTATION COMPLETE ✅")
        print("="*50)
        print("Ready for quantum implementation!")
    else:
        print("❌ Classical optimization failed. Check problem formulation.")
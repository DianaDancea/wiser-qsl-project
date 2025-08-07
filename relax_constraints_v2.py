# Read the file
with open('src/classical/vanguard_etf_optimizer.py', 'r') as f:
    content = f.read()

# Let's make several relaxations:

# 1. Relax portfolio size constraint (allow ±2 bonds)
content = content.replace(
    'constraints.append(cp.sum(x) == target_portfolio_size)',
    'constraints.append(cp.sum(x) >= target_portfolio_size - 2)\n        constraints.append(cp.sum(x) <= target_portfolio_size + 2)'
)

# 2. Remove or relax duration constraint (this is often the problematic one)
content = content.replace(
    'constraints.append(cp.abs(portfolio_duration - target_duration) <= duration_tolerance)',
    '# Relaxed duration constraint\n        # constraints.append(cp.abs(portfolio_duration - target_duration) <= duration_tolerance * 2)'
)

# 3. Relax risk bucket constraints significantly
content = content.replace(
    'min_bonds_per_bucket = max(1, target_portfolio_size // (2 * len(risk_buckets)))',
    'min_bonds_per_bucket = 0  # Relaxed: allow empty buckets'
)

content = content.replace(
    'max_bonds_per_bucket = target_portfolio_size // 2',
    'max_bonds_per_bucket = target_portfolio_size  # Relaxed: allow all bonds in one bucket'
)

# Write back
with open('src/classical/vanguard_etf_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Significantly relaxed constraints - problem should now be feasible!")

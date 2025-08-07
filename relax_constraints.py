# Read the file
with open('src/classical/vanguard_etf_optimizer.py', 'r') as f:
    content = f.read()

# Find and relax some constraints
# Let's relax the risk bucket constraints from exact to approximate
content = content.replace(
    'constraints.append(cp.sum(x[bucket_bonds]) >= min_bonds_per_bucket)',
    'constraints.append(cp.sum(x[bucket_bonds]) >= max(1, min_bonds_per_bucket - 1))'
)

content = content.replace(
    'constraints.append(cp.sum(x[bucket_bonds]) <= max_bonds_per_bucket)',
    'constraints.append(cp.sum(x[bucket_bonds]) <= max_bonds_per_bucket + 1)'
)

# Also relax the duration constraint slightly
content = content.replace(
    'constraints.append(cp.abs(portfolio_duration - target_duration) <= duration_tolerance)',
    'constraints.append(cp.abs(portfolio_duration - target_duration) <= duration_tolerance * 1.5)'
)

# Write back
with open('src/classical/vanguard_etf_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Relaxed constraints to make problem more feasible!")

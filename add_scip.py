# Read the file
with open('src/classical/vanguard_etf_optimizer.py', 'r') as f:
    content = f.read()

# Replace the solver list to include SCIP first
old_line = "        solvers_to_try = ['CLARABEL', 'ECOS_BB', 'GLPK_MI', 'CVXOPT']"
new_line = "        solvers_to_try = ['SCIP', 'CLARABEL', 'ECOS_BB', 'GLPK_MI', 'CVXOPT']"

content = content.replace(old_line, new_line)

# Write back
with open('src/classical/vanguard_etf_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Added SCIP as the first solver to try!")

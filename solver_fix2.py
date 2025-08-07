# Read the file
with open('src/classical/vanguard_etf_optimizer.py', 'r') as f:
    content = f.read()

# Replace the solver list
old_line = "        solvers_to_try = ['ECOS_BB', 'CVXOPT', 'SCS']"
new_line = "        solvers_to_try = ['CLARABEL', 'ECOS_BB', 'GLPK_MI', 'CVXOPT']"

content = content.replace(old_line, new_line)

# Write back
with open('src/classical/vanguard_etf_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Updated solver list to include CLARABEL and GLPK_MI!")

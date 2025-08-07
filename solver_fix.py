import re

# Read the file
with open('src/classical/vanguard_etf_optimizer.py', 'r') as f:
    content = f.read()

# Replace the problematic line
old_line = '        problem.solve(verbose=True)'
new_lines = '''        # Try solvers in order of preference
        solvers_to_try = ['ECOS_BB', 'CVXOPT', 'SCS']
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
            raise Exception("All solvers failed. Problem may be infeasible.")'''

# Replace the line
content = content.replace(old_line, new_lines)

# Write back
with open('src/classical/vanguard_etf_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Solver fix applied!")

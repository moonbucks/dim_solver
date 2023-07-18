from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('Minimize Cost', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# gpt2 config
world_size = 4
batch_size = 12 
num_layers = 12

n_stage = solver.IntVar(1, solver.infinity(), 'Pipeline depth')
n_chunk = solver.IntVar(1, solver.infinity(), 'Number of chunks')
i_stage = solver.IntVar(1, solver.infinity(), 'Internal pipeline depth')


# constraints
solver.Add(n_stage <= world_size)
solver.Add(n_chunk <= batch_size)
solver.Add(i_stage <= 2 * num_layers / n_stage) # Warning multiplication
#solver.Add(i_stage <= 2 * num_layers) # Warning Right value not correct 
solver.Add(n_chunk >= 4 * n_stage) # recommended 
solver.Add(n_chunk >= i_stage - 1) # not a "requirement"

solver.Maximize(n_chunk)

status = solver.Solve()

# If an optimal solution has been found, print results
if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
  print('================= Solution =================')
  print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
  print(f'Optimal value = {solver.Objective().Value()} ')
  print(f' - Pipeline Depth = {n_stage.solution_value()}')
  print(f' - Number of Chunks = {n_chunk.solution_value()}')
  print(f' - Internal Pipeine Depth = {i_stage.solution_value()}')
else:
  print('The solver could not find an optimal solution.')


# Reference
# https://towardsdatascience.com/introduction-to-linear-programming-in-python-9261e7eb44b

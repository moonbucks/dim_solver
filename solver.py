from ortools.linear_solver import pywraplp
from cost import pipeline_cost, pipeline_cost_pippy

def solve_linear(args, n_stage, tp):
  solver = pywraplp.Solver('Minimize Cost', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

  ## variables 
  size_microbatch = solver.IntVar(1, solver.infinity(), 'Size of microbatch')
  i_stage = solver.IntVar(1, solver.infinity(), 'Internal pipeline depth')

  ## constraints
  solver.Add(size_microbatch <= args.batch_size)
  solver.Add(i_stage <= 2 * args.num_layers / n_stage)
  solver.Add(size_microbatch >= i_stage) 

  ## objective 
  cost = pipeline_cost(args, n_stage, size_microbatch, i_stage)

  ## solve
  solver.Minimize(cost)
  status = solver.Solve()

  ## results 
  if status == pywraplp.Solver.OPTIMAL: # or status == pywraplp.Solver.FEASIBLE:
    print(f'================= Solution for {[n_stage, tp]} =================')
    print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
    print(f' - Optimal Cost = {solver.Objective().Value()} ')
    print(f' - Size of Microbatch = {size_microbatch.solution_value()}')
    print(f' - Internal Pipeine Depth = {i_stage.solution_value()}')
    print()

    return solver.Objective().Value(), size_microbatch.solution_value(), i_stage.solution_value()

  else:
    print(f'================= Solution for {[n_stage, tp]} =================')
    print('The solver could not find an optimal solution.')
    print()

    return float('inf'), -1, -1


def solve_pippy(args, n_stage, tp):
  solver = pywraplp.Solver('Minimize Cost', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

  ## variables 
  size_microbatch = solver.IntVar(1, solver.infinity(), 'Size of microbatch')
  i_stage = solver.IntVar(1, solver.infinity(), 'Internal pipeline depth')

  ## constraints 
  solver.Add(size_microbatch <= args.batch_size)
  solver.Add(i_stage <= 2 * args.num_layers // n_stage)
  solver.Add(size_microbatch >= i_stage) 

  ## objective 
  cost = pipeline_cost_pippy(args, n_stage, size_microbatch, i_stage)

  ## solve
  solver.Minimize(cost)
  status = solver.Solve()

  ## results 
  if status == pywraplp.Solver.OPTIMAL: # or status == pywraplp.Solver.FEASIBLE:
    print(f'================= Solution for {[n_stage, tp]} =================')
    print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
    print(f' - Optimal Cost = {solver.Objective().Value()} ')
    print(f' - Size of Microbatch = {size_microbatch.solution_value()}')
    print(f' - Internal Pipeine Depth = {i_stage.solution_value()}')
    print()

    return solver.Objective().Value(), size_microbatch.solution_value(), i_stage.solution_value()

  else:
    print(f'================= Solution for {[n_stage, tp]} =================')
    print('The solver could not find an optimal solution.')
    print()

    return float('inf'), -1, -1

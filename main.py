from ortools.linear_solver import pywraplp
from util import get_all_possible_dims, get_args, set_model_args
from cost_linear import pipeline_cost

def solve(args, n_stage, tp):
  solver = pywraplp.Solver('Minimize Cost', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

  ## variables 
  #n_chunk = solver.IntVar(1, solver.infinity(), 'Number of chunks')
  size_microbatch = solver.IntVar(1, solver.infinity(), 'Size of microbatch')
  i_stage = solver.IntVar(1, solver.infinity(), 'Internal pipeline depth')

  ## constraints
  solver.Add(size_microbatch <= args.batch_size)
  solver.Add(i_stage <= 2 * args.num_layers / n_stage)
  #solver.Add(n_chunk >= 4 * n_stage) # recommended since it is a 'recommended' thing, it is actually not a 'constraint'. this can be relaxed  
  solver.Add(size_microbatch >= i_stage - 1) # preferred to hide all internal ARs actually this is more likely 

  ## objective 

  #toy
  #cost = toy_cost(n_chunk)

  #real
  cost = pipeline_cost(args, n_stage, size_microbatch, i_stage)

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

if __name__ == '__main__':
  args = get_args()
  set_model_args(args, 'gpt3-175b')
  lst = get_all_possible_dims(args.num_gpu)
  best_cost = float('inf')
  best_dim = [1,1]
  for dim in lst:
    if dim[0] > args.num_layers: # assert pp_size <= world_size
      continue

    cost, size_microbatch, i_stage = solve(args, dim[0], dim[1])
    if cost < best_cost:
      best_cost = cost
      best_dim = dim
      best_size_microbatch = size_microbatch
      best_i_stage = i_stage

  print(best_dim, best_cost, 'size_microbatch:', size_microbatch, 'internal stage:', best_i_stage)

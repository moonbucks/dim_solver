from ortools.linear_solver import pywraplp
from utils import get_all_possible_dims, get_args, set_model_args
import solver

def solve(args, n_stage, tp):
  if args.cost_function == 'ideal':
    return solver.solve_linear(args, n_stage, tp)
  else:
    return solver.solve_pippy(args, n_stage, tp)

if __name__ == '__main__':
  args = get_args()
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

  print(best_dim, best_cost, 'size_microbatch:', best_size_microbatch, 'internal stage:', best_i_stage)

  # output data: pp_group_size tp_group_size size_microbatch i_stage n_chunk
  n_chunk = args.batch_size // best_size_microbatch
  f = open('./solver.out', 'w')
  f.write(f'{best_dim[0]} {best_dim[1]} {int(best_size_microbatch)} {int(best_i_stage)} {int(n_chunk)}')
  f.close()

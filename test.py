from cost_linear import stage_cost, pp_comm_cost, tp_comm_cost
from utils import get_args

size_microbatch = 12 
pp_n_chunks = 8
batch_size = size_microbatch * pp_n_chunks
i_stage = 1

def print_cost(args, pp_group_size, tp_group_size):
  _stage_cost = stage_cost(args, pp_group_size, i_stage)
  comp_cost = _stage_cost * batch_size 
  _tp_comm_cost = tp_comm_cost(args, pp_group_size, i_stage) 
  _pp_comm_cost = pp_comm_cost(args) * size_microbatch * (pp_group_size - 1)
  bubble_overhead = stage_cost(args, pp_group_size, i_stage) * size_microbatch * (pp_group_size -1)

  print(f'PP{pp_group_size}xTP{tp_group_size}= Stage {_stage_cost}, Comp {comp_cost}, TPComm {_tp_comm_cost}, PPComm {_pp_comm_cost}, Bubble {bubble_overhead}') 

  


if __name__ == '__main__':
  args = get_args()
  config = [[1,8], [2,4], [4,2], [8,1]]
  for pp_group_size, tp_group_size in config:
    print_cost(args, pp_group_size, tp_group_size)

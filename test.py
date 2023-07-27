from cost_linear import stage_cost, pp_comm_cost 
from utils import get_args

size_microbatch = 1  
def print_cost(args, pp_group_size, tp_group_size):
  _stage_cost = stage_cost(args, pp_group_size, 1)
  comp_cost = _stage_cost * 512
  _pp_comm_cost = pp_comm_cost(args) * size_microbatch * (pp_group_size - 1)
  bubble_overhead = stage_cost(args, pp_group_size, 1) * size_microbatch * (pp_group_size -1)

  print(f'PP{pp_group_size}xTP{tp_group_size}= Stage{_stage_cost}, Comp{comp_cost}, PPComm{_pp_comm_cost}, Bubble{bubble_overhead}') 

  


if __name__ == '__main__':
  args = get_args()
  config = [[8,1], [4,2], [2,4], [1,8]]
  for pp_group_size, tp_group_size in config:
    print_cost(args, pp_group_size, tp_group_size)

import math
import sys

# here we are only calculating forward path's cost
def pipeline_cost(args, n_stage, n_chunk, i_stage):
  # TODO add assertion. if false, return float('inf')

  #n_microbatch = args.batch_size // n_chunk # WARNING non-linear
  n_microbatch = n_chunk # alternative. but need to be fixed
  cost = ( stage_cost(args, n_stage) * n_microbatch   # computation 
          + (n_microbatch - 1) * pp_comm_cost(args) # pp communication
          + stage_cost(args, n_stage) * (n_stage - 1) # bubble
          )

  return cost

def stage_cost(args, n_stage):
  #if args.num_layers % n_stage != 0:
  #  print(f'[stage_cost] {args.num_layers} % {n_stage} = {args.num_layers % n_stage}')
  #  return float('inf')

  num_layers_per_stage = math.ceil(args.num_layers / n_stage) # TODO double check

  ### compute cost
  attn_flops = ( 4 * args.block_size * args.num_embd * args.num_embd 
                + 2 * args.block_size * args.block_size * args.num_embd)
  mlp_flops = (4 * args.num_embd * args.num_embd * args.block_size * 2)

  perf = args.compute_performance * pow(10, 12)

  comp_cost = num_layers_per_stage * (attn_flops + mlp_flops) / perf

  ### tp communication cost

  tp_size = args.num_gpu // n_stage

  bandwidth = args.network_bandwidth * pow(10, 9)

  # TODO replace 4 to sizeof(float)
  attn_comm = 0 / tp_size * 4 / bandwidth
  mlp_comm = 0 / tp_size * 4 / bandwidth

  # we assume AR1 optimization
  tp_comm_cost = (attn_comm * (2 * num_layers_per_stage - 1) 
                  + mlp_comm * 2 * num_layers_per_stage ) 

  cost = comp_cost + tp_comm_cost

  return cost

def tp_comm_cost(args):
  cost = 0

  return cost

def pp_comm_cost(args):
  cost = 0

  return cost

def toy_cost(n_chunk):
  return -1 * n_chunk

import math

# here we are only calculating forward path's cost
def pipeline_cost(args, n_stage, size_microbatch, i_stage):
  # TODO add assertion. if false, return float('inf')

  cost = ( stage_cost(args, n_stage, i_stage) * args.batch_size # computation 
          + pp_comm_cost(args) * size_microbatch * (n_stage - 1) # pp communication
          + stage_cost(args, n_stage, i_stage) * size_microbatch * (n_stage - 1) # bubble WARNING non-linear (multiplication when --opt-ar0 enabled)
          )

  return cost

def stage_cost(args, n_stage, i_stage):
  #if args.num_layers % n_stage != 0:
  #  print(f'[stage_cost] {args.num_layers} % {n_stage} = {args.num_layers % n_stage}')
  #  return float('inf')

  num_layers_per_stage = math.ceil(args.num_layers / n_stage)

  ### compute cost
  attn_flops = ( 4 * args.block_size * args.num_embd * args.num_embd 
                + 2 * args.block_size * args.block_size * args.num_embd)
  mlp_flops = (4 * args.num_embd * args.num_embd * args.block_size * 2)

  perf = args.compute_performance * pow(10, 12)

  comp_cost = num_layers_per_stage * (attn_flops + mlp_flops) / perf

  ### tp communication cost
  tp_size = args.num_gpu // n_stage

  # we assume single precision parameters
  bandwidth = args.network_bandwidth * pow(10, 9)

  # tensor size after attn_c_proj: [batch, block_size, embd_size]
  # batch is multiplied outside this function
  attn_comm = args.block_size * args.num_embd / tp_size * 4 / bandwidth

  # tensor size after mlp_c_proj: [batch, block_size, embd_size]
  # batch is multiplied outside this function
  mlp_comm = args.block_size * args.num_embd / tp_size * 4 / bandwidth

  # depending on optimization level, tp cost varies 
  if args.opt_ar1 and args.opt_ar0:
    tp_comm_cost = (attn_comm * (num_layers_per_stage - 1) 
                    + mlp_comm * num_layers_per_stage 
                    - max(attn_comm, mlp_comm) * (i_stage-1))
  elif args.opt_ar1:
    tp_comm_cost = (attn_comm * (num_layers_per_stage - 1) 
                    + mlp_comm * num_layers_per_stage) 
  else:
    tp_comm_cost = (attn_comm + mlp_comm) * num_layers_per_stage

  cost = comp_cost + tp_comm_cost 

  return cost

def pp_comm_cost(args):
  # tensor size after mlp_c_proj: [batch, block_size, embd_size] 
  # batch is multiplied outside this function (L9)
  bandwidth = args.network_bandwidth * pow(10, 9)
  cost = args.block_size * args.num_embd * 4 / bandwidth 
  return cost

def toy_cost(n_chunk):
  return -1 * n_chunk

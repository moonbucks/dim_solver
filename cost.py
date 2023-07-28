import math

# here we compare with forward path's cost
def pipeline_cost(args, n_stage, size_microbatch, i_stage):
  if args.cost_function == 'ideal':
    cost = pipeline_cost_ideal(args, n_stage, size_microbatch, i_stage)
  else:
    cost = pipeline_cost_custom(args, n_stage, size_microbatch, i_stage)
  return cost

def pipeline_cost_ideal(args, n_stage, size_microbatch, i_stage):
  cost = ( stage_cost(args, n_stage, i_stage) * args.batch_size # computation 
          + pp_comm_cost(args) * size_microbatch * (n_stage - 1) # pp communication of bubble stages
          + stage_cost(args, n_stage, i_stage) * size_microbatch * (n_stage - 1) # bubble 
          )

  return cost

def pipeline_cost_custom(args, n_stage, size_microbatch, i_stage):
  raise NotImplementedError("Define your own cost function")

  return float('inf')

def pipeline_cost_pippy(args, n_stage, size_microbatch, i_stage):
  assert args.bubble_cost > 0 and args.stage_cost > 0 and args.tp_cost > 0 and args.pp_cost > 0, "Pass your profiled value"

  # obtain following cost with microbatch-size=1 
  # args.bubble_cost: waiting time for computation of previous stages; 
  #                   includes pp cost, in our case it is around 200ms 
  # args.stage_cost: from the start of first operation to the end of last computation op 
  # args.tp_cost: last all reduce time, in our case 0.05ms per microbatch
  # args.pp_cost: 3ms per microbatch 

  n_chunks = args.batch_size // size_microbatch 

  num_layers_per_stage = math.ceil(args.num_layers / n_stage)
  _forward_cost = ( args.stage_cost +  args.tp_cost ) * n_chunks
  _tp_cost = args.tp_cost * size_microbatch
  _bubble_cost = (args.stage_cost + args.pp_cost * size_microbatch) * (n_stage - 1)

  if args.opt_ar1 and args.opt_ar0:
    _forward_cost -= _tp_cost * i_stage * n_chunks 
  elif args.opt_ar1:
    _forward_cost -= _tp_cost * n_chunks 

  cost = _bubble_cost + _forward_cost 

  return cost

def stage_cost(args, n_stage, i_stage):
  ### compute cost
  num_layers_per_stage = math.ceil(args.num_layers / n_stage)
  attn_flops = ( 4 * args.block_size * args.num_embd * args.num_embd 
                + 2 * args.block_size * args.block_size * args.num_embd)
  mlp_flops = (4 * args.num_embd * args.num_embd * args.block_size * 2)
  perf = args.compute_performance * pow(10, 12)
  comp_cost = num_layers_per_stage * (attn_flops + mlp_flops) / perf
  cost = comp_cost + tp_comm_cost(args, n_stage, i_stage) 
  return cost

def pp_comm_cost(args):
  # tensor size after mlp_c_proj: [batch, block_size, embd_size] 
  # batch is multiplied outside this function
  bandwidth = args.network_bandwidth * pow(10, 9)
  cost = args.block_size * args.num_embd * 4 / bandwidth 
  return cost

def tp_comm_cost(args, n_stage, i_stage):
  num_layers_per_stage = math.ceil(args.num_layers / n_stage)
  ### tp communication cost
  tp_size = args.num_gpu // n_stage
  # we assume single precision parameters
  bandwidth = args.network_bandwidth * pow(10, 9)

  # tensor size after attn_c_proj: [batch, block_size, embd_size]
  # batch is multiplied outside this function
  attn_comm = args.block_size * args.num_embd * (tp_size - 1) / tp_size * 4 / bandwidth

  # tensor size after mlp_c_proj: [batch, block_size, embd_size]
  # batch is multiplied outside this function
  mlp_comm = args.block_size * args.num_embd * (tp_size - 1) / tp_size * 4 / bandwidth

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

  return tp_comm_cost

def toy_cost(n_chunk):
  return -1 * n_chunk

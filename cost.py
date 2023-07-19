def pipeline_cost(args, n_chunk, i_stage):
  cost = 0 

  return cost

def comp_cost(args):
  cost = 0

  return cost

def comm_cost(args):
  cost = pp_comm_cost(args) + tp_comm_cost(args) 

  return cost

def pp_comm_cost(args):
  cost = 0

  return cost

def tp_comm_cost(args):
  cost = 0

  return cost

def toy_cost(n_chunk):
  return -1 * n_chunk

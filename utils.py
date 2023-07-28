import argparse
import os

def get_args(extra_args_provider=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-gpu', type=int, default=8)

  # system
  # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
  parser.add_argument('--network-bandwidth', type=float, default=600, help='network bandwith in GB/s') # default=nvlink
  parser.add_argument('--compute-performance', type=float, default=19.5, help='single-precision floating point performance in TFLOPS') # default=A100 

  # model (default: gpt-2)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--num-layers', type=int, default=12)
  parser.add_argument('--num-head', type=int, default=12)
  parser.add_argument('--num-embd', type=int, default=768)
  parser.add_argument('--block-size', type=int, default=1024)
  parser.add_argument('--vocab-size', type=int, default=50304)

  # optimization
  parser.add_argument('--opt-ar1', action='store_true', help='optimizing the last ar')
  parser.add_argument('--opt-ar0', action='store_true', help='optimizing all inner ars')

  # profile args
  parser.add_argument('--cost-function', type=str, default='ideal', help='predefined: ideal, profiled')
  parser.add_argument('--stage-cost', type=float, default=0.0, help='in ms, including tp cost')
  parser.add_argument('--pp-cost', type=float, default=0.0, help='unit pp cost in ms')
  parser.add_argument('--tp-cost', type=float, default=0.0, help='unit tp cost in ms')
  parser.add_argument('--bubble-cost', type=float, default=0.0, help='bubble cost in ms')

  args = parser.parse_args()
  return args

def set_model_args(args, model_name='gpt3-175b'):
  if model_name == 'gpt3-175b':
    args.batch_size = 16384
    args.num_layers = 96
    args.num_head = 96
    args.num_embd = 128
    args.block_size = 2048
  else:
    print('model config does not exist.')

  return args

def get_all_possible_dims (num_gpu):
  # TODO support non power-of-two numbers
  lst = [[num_gpu,1]]
  tp, pp = 1, num_gpu
  while pp>1:
    pp /=2
    tp *=2
    lst.append([int(pp),int(tp)])

  print(f'num_gpu:{num_gpu}, len(lst)={len(lst)}, lst={lst}')
  return lst


if __name__ == '__main__':
  get_all_possible_dims(16)
  get_all_possible_dims(32)
  get_all_possible_dims(64)
  get_all_possible_dims(256)
  get_all_possible_dims(512)
  get_all_possible_dims(1024)
  get_all_possible_dims(2048)
  get_all_possible_dims(1008)
  get_all_possible_dims(1052)

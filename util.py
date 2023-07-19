import argparse
import os

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-gpu', type=int, default=4)

  # system
  parser.add_argument('--network-bandwidth', type=float, default=300, help='network bandwith in GB/s') # default=nvlink
  parser.add_argument('--compute-performance', type=float, default=19.5, help='single-precision floating point performance in TFLOPS') # default=A100 

  # model
  parser.add_argument('--num-layers', type=int, default=12)
  parser.add_argument('--batch-size', type=int, default=12)

  args = parser.parse_args()
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

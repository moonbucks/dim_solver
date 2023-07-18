def get_all_possible_dims (num_gpu):
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

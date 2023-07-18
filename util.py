def possible_dim (num_gpu):

  lst = [[num_gpu,1]]
  tp, pp = 1, num_gpu
  while pp>1:
    pp /=2
    tp *=2
    lst.append([int(pp),int(tp)])

  print(f'num_gpu:{num_gpu}, len(lst)={len(lst)}, lst={lst}')
  return lst


if __name__ == '__main__':
  possible_dim(16)
  possible_dim(32)
  possible_dim(64)
  possible_dim(256)
  possible_dim(512)
  possible_dim(1024)
  possible_dim(2048)

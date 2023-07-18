# dim_solver

dimension solver for 2d (pp+tp) parallelism

## Install and Run

### Install Dependencies

```
pip install ortools
```

### Run
```
python main.py --num-gpu 4 --network-bandwith 300 --compute-performance 100
```

parameters: 

- num_gpu= world size
- network_bandwidth= network bandwidth in GB/s
- compute_performance= single precision performance in FLOPs/s

model config will be added later. 

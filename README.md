# Dimension solver

This is a dimension solver for automated 2d (pp+tp) parallelism. 

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
- compute_performance= single precision performance in TFLOPs
- batch_size= batch size
- num_layers= number of transformer blocks
- num_head= number of head
- num_embd= embedding size
- block_size= sequence length
- opt_ar1= when the last AR is hidden
- opt_ar0= when inner layers are hidden

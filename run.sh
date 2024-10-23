#!/bin/bash

# bert
# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --random_factor --report_dir_postfix random_factor
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --random_factor --report_dir_postfix random_factor
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --random_factor --report_dir_postfix random_factor
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --random_factor --report_dir_postfix random_factor
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --random_factor --report_dir_postfix random_factor
# cd /root/project/Soter

# ====================================================================================
# random vs soter (long epoch 100)
# ====================================================================================
# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --random_factor --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --random_factor --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --random_factor --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --random_factor --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --random_factor --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# ====================================================================================
# random vs soter (tensor core)
# ====================================================================================
# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 100 --accelerator TensorCore --workload bertlarge --layer_id 0 --batch_size 1 --report_dir_postfix ep100
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 100 --accelerator TensorCore --workload bertlarge --layer_id 0 --batch_size 1 --random_factor --report_dir_postfix ep100_random
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 50 --accelerator TensorCore --workload bertlarge --layer_id 1 --batch_size 1 --report_dir_postfix ep50
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 50 --accelerator TensorCore --workload bertlarge --layer_id 1 --batch_size 1 --random_factor --report_dir_postfix ep50_random
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 50 --accelerator TensorCore --workload bertlarge --layer_id 4 --batch_size 1 --report_dir_postfix ep50
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 50 --accelerator TensorCore --workload bertlarge --layer_id 4 --batch_size 1 --random_factor --report_dir_postfix ep50_random
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 50 --accelerator TensorCore --workload bertlarge --layer_id 5 --batch_size 1 --report_dir_postfix ep50
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 50 --accelerator TensorCore --workload bertlarge --layer_id 5 --batch_size 1 --random_factor --report_dir_postfix ep50_random
# cd /root/project/Soter

# ====================================================================================
# random vs soter (tensor core with super long)
# ====================================================================================
# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 0 --batch_size 1 --report_dir_postfix ep1000
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 0 --batch_size 1 --random_factor --report_dir_postfix ep1000_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 3 --batch_size 1 --report_dir_postfix ep1000
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 3 --batch_size 1 --random_factor --report_dir_postfix ep1000_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 4 --batch_size 1 --report_dir_postfix ep1000
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 4 --batch_size 1 --random_factor --report_dir_postfix ep1000_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 5 --batch_size 1 --report_dir_postfix ep1000
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 5 --batch_size 1 --random_factor --report_dir_postfix ep1000_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# ================================================================================
# Tensor Core
# ================================================================================
layer_ids=(0 3 4 5)
epochs=(100 1000)
mode=(origin random)

for layer_id in "${layer_ids[@]}"; do
  for epoch in "${epochs[@]}"; do
    for m in "${mode[@]}"; do
      if [[ $m == "origin" ]]; then
        flag=""
      elif [[ $m == "random" ]]; then
        flag="--random_factor --random_order"
      fi
      CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs $epoch --accelerator TensorCore \
        --workload bertlarge --layer_id $layer_id --batch_size 1 \
        --report_dir_postfix ep${epoch}_${m} \
        $flag
      cd /root/project/Soter
      find /dev/shm -type f | xargs rm -f
    done

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 100 --accelerator TensorCore --workload bertlarge --layer_id 0 --batch_size 1 --random_factor --random_order --report_dir_postfix ep100_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 100 --accelerator TensorCore --workload bertlarge --layer_id 3 --batch_size 1 --random_factor --random_order --report_dir_postfix ep100_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 100 --accelerator TensorCore --workload bertlarge --layer_id 4 --batch_size 1 --random_factor --random_order --report_dir_postfix ep100_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 100 --accelerator TensorCore --workload bertlarge --layer_id 5 --batch_size 1 --random_factor --random_order --report_dir_postfix ep100_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f
# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 0 --batch_size 1 --random_factor --random_order --report_dir_postfix ep1000_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 3 --batch_size 1 --random_factor --random_order --report_dir_postfix ep1000_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 4 --batch_size 1 --random_factor --random_order --report_dir_postfix ep1000_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 1000 --accelerator TensorCore --workload bertlarge --layer_id 5 --batch_size 1 --random_factor --random_order --report_dir_postfix ep1000_true_random
# cd /root/project/Soter
# find /dev/shm -type f | xargs rm -f
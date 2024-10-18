# bert
# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --random_sample --report_dir_postfix random_sample
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --random_sample --report_dir_postfix random_sample
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --random_sample --report_dir_postfix random_sample
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --random_sample --report_dir_postfix random_sample
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --report_dir_postfix origin
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=2 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --random_sample --report_dir_postfix random_sample
# cd /root/project/Soter

# ====================================================================================
# random vs soter (long epoch 100)
# ====================================================================================
CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --report_dir_postfix long_search
cd /root/project/Soter

CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1 --random_sample --report_dir_postfix long_search_random_search
cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1 --random_sample --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1 --random_sample --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1 --random_sample --report_dir_postfix long_search_random_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --report_dir_postfix long_search
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj edp --epochs 100 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1 --random_sample --report_dir_postfix long_search_random_search
# cd /root/project/Soter
#CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 10 --accelerator Simba --workload resnet50 --layer_id 43 --batch_size 1
#CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 16
#CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload gpt3 --layer_id 3 --batch_size 16
#CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Eyeriss --workload gpt3 --layer_id 3 --batch_size 16
# CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload gpt3 --layer_id 0 --batch_size 16
# CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload gpt1 --layer_id 0 --batch_size 8

# CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 0 --batch_size 1
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 1 --batch_size 1
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 2 --batch_size 1
# cd /root/project/Soter

# CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 3 --batch_size 1
# cd /root/project/Soter

CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj edp --epochs 30 --accelerator Simba --workload bertlarge --layer_id 4 --batch_size 1
cd /root/project/Soter

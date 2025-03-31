# SESSION_NAME=vmalv
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=0
# python train_id.py --lr 1e-4 --weight_decay 1e-5 --hidden_size 4096 --num_layers 5
# " C-m

# SESSION_NAME=vmalv1
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=1
# python train_id.py --lr 1e-4 --weight_decay 1e-4 --hidden_size 4096 --num_layers 5
# " C-m
# #!/bin/bash

# Hyperparameter grids
lrs=(5e-4 1e-3 1e-4)
weight_decays=(0)
hidden_sizes=(128 256 512 1024) # 256 1024 2048 4096
num_layers=(4)
frame_stack=(2)

# GPU configuration
gpus=(0 1 2 3 4 5 6 7)
experiments_per_gpu=4

# Total experiments
total_experiments=32
experiment=0

# Kill any existing tmux server
tmux kill-server

for lr in "${lrs[@]}"; do
    for weight_decay in "${weight_decays[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for layer in "${num_layers[@]}"; do
                for stack in "${frame_stack[@]}"; do
                    if [ $experiment -ge $total_experiments ]; then
                        break 4
                    fi

                    gpu=${gpus[$((experiment / experiments_per_gpu))]}
                    SESSION_NAME="exp_${experiment}"

                    tmux new-session -d -s $SESSION_NAME
                    tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
                    tmux send-keys -t $SESSION_NAME "
                    export CUDA_VISIBLE_DEVICES=${gpu}
                    python train_id.py --lr ${lr} --weight_decay ${weight_decay} --hidden_dim ${hidden_size} --num_layers ${layer} --frame_stack ${stack}
                    " C-m

                    echo "Launched ${SESSION_NAME} on GPU ${gpu} with lr=${lr}, weight_decay=${weight_decay}, hidden_size=${hidden_size}, num_layers=${layer}"
                    experiment=$((experiment + 1))
                done
            done
        done
    done
done
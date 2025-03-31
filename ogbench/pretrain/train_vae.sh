# SESSION_NAME=0
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=0
# python train_vae.py --dataset_url /home/hyeons/workspace/ogbench/data_gen_scripts/data/visual-pointmaze-large-navigate-v0.npz
# " C-m

# SESSION_NAME=1
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=1
# python train_vae.py --dataset_url /home/hyeons/workspace/ogbench/data_gen_scripts/data/visual-pointmaze-gaint-navigate-v0.npz
# " C-m
SESSION_NAME=4
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=4
python train_vae.py --dataset_url /home/hyeons/workspace/ogbench/data_gen_scripts/data/visual-pointmaze-large-navigate-v0.npz --load /home/hyeons/workspace/ogbench/ogbench/pretrain/large_100.pth
" C-m

SESSION_NAME=6
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=6
python train_vae.py --dataset_url /home/hyeons/workspace/ogbench/data_gen_scripts/data/visual-pointmaze-giant-navigate-v0.npz --load /home/hyeons/workspace/ogbench/ogbench/pretrain/giant_100.pth
" C-m


# SESSION_NAME=3
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
# tmux send-keys -t $SESSION_NAME "
# export CUDA_VISIBLE_DEVICES=0
# python train_e2s_id.py --dataset_url /home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/large --mode invd
# " C-m

SESSION_NAME=2
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=0
python train_e2s_id.py --dataset_url /home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/large --mode e2s
" C-m


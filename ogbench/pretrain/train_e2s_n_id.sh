SESSION_NAME=0
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=0
python train_id.py --dataset_url /home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/large 
" C-m

SESSION_NAME=1
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=1
python train_id.py --dataset_url /home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/giant
" C-m

SESSION_NAME=2
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=2
python train_emb2state.py --dataset_url /home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/large 
" C-m

SESSION_NAME=3
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "conda activate og_game" C-m
tmux send-keys -t $SESSION_NAME "
export CUDA_VISIBLE_DEVICES=3
python train_emb2state.py --dataset_url /home/hyeons/workspace/ogbench/ogbench/pretrain/embedded_data/giant
" C-m



export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_PATH_GT=./data/roberta_data_gt_25
export DATA_PATH_SOP=./data/roberta_data_sop_25
export SAVE_PATH=./experiments/
export MODEL_PATH=./roberta.base.tar.gz


python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_base_experiment_5e-5 \
  --model_type roberta \
  --task GT \
  --model $MODEL_PATH \
  --data_dir $DATA_PATH_GT \
  --save_dir $SAVE_PATH \
  --batch_size 10 \
  --accumulation_steps 13 \
  --num_epochs 1 \
  --num_workers 1 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00005 \
  --warmup_proportion 0.1 \
  --max_steps 25000 \
  --fp16 True \


python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_base_experiment_5e-5_sop \
  --model_type roberta \
  --task SOP \
  --model $MODEL_PATH \
  --data_dir $DATA_PATH_SOP \
  --save_dir $SAVE_PATH \
  --batch_size 10 \
  --accumulation_steps 13 \
  --num_epochs 1 \
  --num_workers 1 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00005 \
  --warmup_proportion 0.1 \
  --max_steps 25000 \
  --fp16 True \

python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_base_experiment_5e-5_win0 \
  --model_type roberta \
  --task GT \
  --model $MODEL_PATH \
  --data_dir $DATA_PATH_GT \
  --save_dir $SAVE_PATH \
  --batch_size 10 \
  --accumulation_steps 13 \
  --num_epochs 1 \
  --num_workers 1 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00005 \
  --warmup_proportion 0.1 \
  --max_steps 25000 \
  --fp16 True \
  --window_size 0 \

python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_base_experiment_5e-5_2l \
  --model_type roberta \
  --task GT \
  --model $MODEL_PATH \
  --data_dir $DATA_PATH_GT \
  --save_dir $SAVE_PATH \
  --batch_size 10 \
  --accumulation_steps 13 \
  --num_epochs 1 \
  --num_workers 1 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00005 \
  --warmup_proportion 0.1 \
  --max_steps 25000 \
  --fp16 True \
  --two_layers True \

python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_base_experiment_5e-5_win5 \
  --model_type roberta \
  --task GT \
  --model $MODEL_PATH \
  --data_dir $DATA_PATH_GT \
  --save_dir $SAVE_PATH \
  --batch_size 10 \
  --accumulation_steps 13 \
  --num_epochs 1 \
  --num_workers 1 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00005 \
  --warmup_proportion 0.1 \
  --max_steps 25000 \
  --fp16 True \
  --window_size 5 \

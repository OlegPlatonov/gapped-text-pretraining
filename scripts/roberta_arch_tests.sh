export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROBERTA_DATA_PATH=./data/roberta_data_30
export SAVE_PATH=./experiments/
export MODEL_PATH=./roberta.base.tar.gz

python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_no_window_pool \
  --model_type roberta \
  --model $MODEL_PATH \
  --data_dir $ROBERTA_DATA_PATH \
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
  --fp16 True \
  --window_size 0



python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_2layers \
  --model_type roberta \
  --model $MODEL_PATH \
  --data_dir $ROBERTA_DATA_PATH \
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
  --fp16 True \
  --two_layers True



python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_2layers_no_window_pool \
  --model_type roberta \
  --model $MODEL_PATH \
  --data_dir $ROBERTA_DATA_PATH \
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
  --fp16 True \
  --two_layers True \
  --window_size 0



python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_window8 \
  --model_type roberta \
  --model $MODEL_PATH \
  --data_dir $ROBERTA_DATA_PATH \
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
  --fp16 True \
  --window_size 8

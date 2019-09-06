export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_PATH=./data/roberta_data_30
export SAVE_PATH=./experiments/
export MODEL_PATH=./roberta.base.tar.gz


python -m torch.distributed.launch --nproc_per_node 4 ./train_gt_ddp.py \
  --name roberta_base_experiment_5e-5 \
  --model_type roberta \
  --model $MODEL_PATH \
  --data_dir $DATA_PATH \
  --save_dir $SAVE_PATH \
  --batch_size 5 \
  --accumulation_steps 26 \
  --num_epochs 1 \
  --num_workers 1 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00005 \
  --warmup_proportion 0.1 \

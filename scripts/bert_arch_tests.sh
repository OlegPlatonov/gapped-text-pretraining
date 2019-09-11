export CUDA_VISIBLE_DEVICES=4,5,6,7
export BERT_DATA_PATH=./data/bert_data
export SAVE_PATH=./experiments/

python train_gt.py \
  --name bert_no_cls_pool \
  --data_dir $BERT_DATA_PATH \
  --save_dir $SAVE_PATH \
  --batch_size 6 \
  --num_epochs 1 \
  --num_workers 8 \
  --max_checkpoints 1 \
  --eval_every 250000 \
  --eval_after_epoch True \
  --learning_rate 0.00002 \
  --warmup_proportion 0.1 \
  --freeze_proportion 0 \
  --accumulation_steps 22 \
  --use_output_head False

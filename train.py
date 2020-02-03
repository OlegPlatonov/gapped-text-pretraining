import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

import os
import json
import yaml
from argparse import ArgumentParser
from collections import OrderedDict

from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from models.bert import RobertaForGappedText
from utils.datasets_gt import GT_Dataset, GT_collate_fn
from utils.utils_gt import CheckpointSaver, AverageMeter, get_logger, get_save_dir, get_num_data_samples


"""
Adapted from https://github.com/chrischute/squad and https://github.com/huggingface/pytorch-pretrained-BERT.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='Experiment name.')
    parser.add_argument('--model',
                        type=str,
                        default='roberta-base',
                        help='Pretrained model name or path.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./experiments',
                        help='Base directory for saving information.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/GT')
    parser.add_argument('--seed',
                        type=int,
                        default=12)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='This is the number of training samples processed together by one GPU. '
                             'The effective batch size (number of training samples processed per one '
                             'optimization step) is equal to batch_size * num_gpus * accumulation_steps.')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--fp16',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to use 16-bit float precision instead of 32-bit')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1)
    parser.add_argument('--max_steps',
                        type=int,
                        default=-1)
    parser.add_argument('--learning_rate',
                        default=1e-5,
                        type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use for training data loader.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=30)
    parser.add_argument('--eval_every',
                        type=int,
                        default=50000,
                        help='Evaluate model after processing this many training samples.')
    parser.add_argument('--eval_after_epoch',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to evaluate model at the end of every epoch.')
    parser.add_argument('--apex_level',
                        default='O1',
                        choices=['O0', 'O1', 'O2', 'O3'],
                        help='Apex optimization level. Only used if fp16 is True.')
    parser.add_argument("--weight_decay",
                        default=0,
                        type=float)
    parser.add_argument("--mask_proportion",
                        default=0,
                        type=float,
                        help='Proportion of words to mask in the input.')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help='Local rank for distributed training. Use -1 for single GPU training')

    args = parser.parse_args()

    return args


def train(args, log, tb_writer):
    log.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))
    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

    device_id = args.local_rank if args.local_rank != -1 else 0
    device = torch.device('cuda', device_id)
    log.warning(f'Using GPU {args.local_rank}.')

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    log.info(f'Total number of GPUs used: {world_size}.')
    log.info(f'Effective batch size: {args.batch_size * world_size * args.accumulation_steps}.')

    num_data_samples, num_unique_data_epochs = get_num_data_samples(args.data_dir, args.num_epochs, log)
    num_optimization_steps = sum(num_data_samples) // world_size // args.batch_size // args.accumulation_steps
    if args.max_steps > 0:
        num_optimization_steps = min(num_optimization_steps, args.max_steps)
    log.info(f'Total number of optimization steps: {num_optimization_steps}.')

    # Set random seed
    log.info(f'Using random seed {args.seed}.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    log.info(f'Loading model {args.model}...')
    model = RobertaForGappedText.from_pretrained(args.model)

    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as file:
            json.dump(model.config.__dict__, file)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)

    # Get saver
    saver = CheckpointSaver(args.save_dir,
                            max_checkpoints=args.max_checkpoints,
                            metric_name='Accuracy',
                            maximize_metric=True,
                            log=log)

    # Get optimizer
    log.info('Creating optimizer...')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=num_optimization_steps,
                                                num_warmup_steps=num_optimization_steps * args.warmup_proportion)

    if args.fp16:
        amp.register_half_function(torch, 'einsum')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_level)

    if args.local_rank != -1:
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)



    sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # Get dev data loader
    dev_data_file = os.path.join(args.data_dir, f'Dev.csv')
    log.info(f'Creating dev dataset from {dev_data_file}...')
    dev_dataset = GT_Dataset(dev_data_file)
    dev_sampler = sampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset,
                             batch_size=args.batch_size,
                             sampler=dev_sampler,
                             num_workers=args.num_workers,
                             collate_fn=lambda batch: GT_collate_fn(batch, args.mask_proportion))


    global_step = 0
    samples_processed = 0

    # Train
    log.info('Training...')
    samples_till_eval = args.eval_every
    for epoch in range(1, args.num_epochs + 1):
        if args.local_rank != -1:
            torch.distributed.barrier()

        # Get train data loader for current epoch
        train_data_file_num = ((epoch - 1) % num_unique_data_epochs) + 1
        train_data_file = os.path.join(args.data_dir, f'Epoch_{train_data_file_num}.csv')
        log.info(f'Creating training dataset from {train_data_file}...')
        train_dataset = GT_Dataset(train_data_file)
        train_sampler = sampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   sampler=train_sampler,
                                   num_workers=args.num_workers,
                                   collate_fn=lambda batch: GT_collate_fn(batch, args.mask_proportion))

        if args.local_rank != -1:
            torch.distributed.barrier()

        log.info(f'Starting epoch {epoch}...')
        model.train()
        model.zero_grad()
        gt_loss_val = 0
        lm_loss_val = 0
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset), disable=args.local_rank not in [-1, 0]) as progress_bar:
            for step, batch in enumerate(train_loader, 1):
                batch = tuple(x.to(device) for x in batch)

                input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps, mask_ids, mask_targets = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                gap_ids=gap_ids,
                                target_gaps=target_gaps,
                                mask_ids=mask_ids,
                                mask_targets=mask_targets)

                current_batch_size = input_ids.shape[0]

                gt_loss, lm_loss = outputs[:2]

                if args.accumulation_steps > 1:
                    gt_loss = gt_loss / args.accumulation_steps
                    lm_loss = lm_loss / args.accumulation_steps

                gt_loss_val += gt_loss.item()
                lm_loss_val += lm_loss.item()

                loss = gt_loss + lm_loss

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                samples_processed += current_batch_size * world_size
                samples_till_eval -= current_batch_size * world_size
                progress_bar.update(current_batch_size * world_size)

                if step % args.accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # Log info
                    current_lr = scheduler.get_lr()[0]
                    progress_bar.set_postfix(epoch=epoch, gt_loss=gt_loss_val, lm_loss=lm_loss_val, step=global_step, lr=current_lr)
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('train/Loss', gt_loss_val, global_step)
                        tb_writer.add_scalar('train/LM_loss', lm_loss_val, global_step)
                        tb_writer.add_scalar('train/LR', current_lr, global_step)
                    gt_loss_val = 0
                    lm_loss_val = 0

                    if global_step == args.max_steps:
                        log.info('Reached maximum number of optimization steps.')
                        break

                    if samples_till_eval <= 0:
                        samples_till_eval = args.eval_every
                        evaluate_and_save(model=model,
                                          optimizer=optimizer,
                                          data_loader=dev_loader,
                                          device=device,
                                          tb_writer=tb_writer,
                                          log=log,
                                          global_step=global_step,
                                          saver=saver,
                                          args=args)

            if args.eval_after_epoch:
                evaluate_and_save(model=model,
                                  optimizer=optimizer,
                                  data_loader=dev_loader,
                                  device=device,
                                  tb_writer=tb_writer,
                                  log=log,
                                  global_step=global_step,
                                  saver=saver,
                                  args=args)


def evaluate_and_save(model, optimizer, data_loader, device, tb_writer, log, global_step, saver, args):
    log.info('Evaluating...')
    results = evaluate_GT(model, data_loader, device)

    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                            for k, v in results.items())
    log.info('Dev {}'.format(results_str))

    if args.local_rank in [-1, 0]:
        log.info('Visualizing in TensorBoard...')
        for k, v in results.items():
            tb_writer.add_scalar('dev/{}'.format(k), v, global_step)

        log.info('Saving checkpoint at step {}...'.format(global_step))
        saver.save(step=global_step,
                   model=model,
                   args=args,
                   metric_val=results['Accuracy'])


def evaluate_GT(model, data_loader, device):
    gt_loss_meter = AverageMeter()
    lm_loss_meter = AverageMeter()
    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1

    model.eval()
    correct_preds = 0
    correct_avna = 0
    zero_preds = 0
    total_preds = 0
    correct_mask_preds = 0
    total_mask_preds = 0
    with torch.no_grad(), tqdm(total=len(data_loader.dataset), disable=args.local_rank not in [-1, 0]) as progress_bar:
        for batch in data_loader:
            batch = tuple(x.to(device) for x in batch)
            input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps, mask_ids, mask_targets = batch
            current_batch_size = input_ids.shape[0]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            gap_ids=gap_ids,
                            target_gaps=target_gaps,
                            mask_ids=mask_ids,
                            mask_targets=mask_targets)

            gt_loss, lm_loss, gap_scores, mask_scores = outputs[:4]
            gt_loss_meter.update(gt_loss.item(), current_batch_size)
            lm_loss_meter.update(lm_loss.item(), mask_scores.shape[0])

            preds = torch.argmax(gap_scores, dim=1)
            correct_preds += torch.sum(preds == target_gaps).item()
            correct_avna += torch.sum((preds > 0) == (target_gaps > 0)).item()
            zero_preds += torch.sum(preds == 0).item()
            total_preds += current_batch_size

            mask_preds = torch.argmax(mask_scores, dim=1)
            correct_mask_preds += torch.sum(mask_preds == mask_targets).item()
            total_mask_preds += mask_preds.shape[0]

            # Log info
            progress_bar.update(current_batch_size * world_size)
            progress_bar.set_postfix(gt_loss=gt_loss_meter.avg, lm_loss=lm_loss_meter.avg)

    model.train()

    results_list = [('Loss', gt_loss_meter.avg),
                    ('Accuracy', correct_preds / total_preds),
                    ('AvNA', correct_avna / total_preds),
                    ('NA_share', zero_preds / total_preds),
                    ('LM_loss', lm_loss_meter.avg),
                    ('LM_accuracy', correct_mask_preds / total_mask_preds)]

    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    args = get_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    if args.local_rank in [-1, 0]:
        args.save_dir = get_save_dir(args.save_dir, args.name, training=True)
        log = get_logger(args.save_dir, args.name, log_file=f'log_0.txt')
        log.info(f'Results will be saved to {args.save_dir}.')
        tb_writer = SummaryWriter(args.save_dir)
    else:
        torch.distributed.barrier()
        args.save_dir = get_save_dir(args.save_dir, args.name, training=True, use_existing_dir=True)
        log = get_logger(args.save_dir, args.name, verbose=False, log_file=f'log_{args.local_rank}.txt')
        tb_writer = None

    if args.local_rank == 0:
        torch.distributed.barrier()

    try:
        train(args, log, tb_writer)
    except:
        log.exception('An error occured...')

    if tb_writer is not None:
        tb_writer.close()

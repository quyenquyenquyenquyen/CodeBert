# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import bleu # Đảm bảo file bleu.py nằm trong cùng thư mục hoặc PYTHONPATH
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq # Đảm bảo model.py nằm trong cùng thư mục hoặc PYTHONPATH
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return examples
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            try:
                js=json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {idx+1} in {filename}: {line}")
                continue

            if 'idx' not in js:
                js['idx']=idx
            
            # Ensure 'code_tokens' and 'docstring_tokens' exist and are lists
            code_tokens = js.get('code_tokens', [])
            if not isinstance(code_tokens, list):
                logger.warning(f"Invalid 'code_tokens' format at line {idx+1} in {filename}. Expected list, got {type(code_tokens)}. Using empty list.")
                code_tokens = []

            docstring_tokens = js.get('docstring_tokens', [])
            if not isinstance(docstring_tokens, list):
                logger.warning(f"Invalid 'docstring_tokens' format at line {idx+1} in {filename}. Expected list, got {type(docstring_tokens)}. Using empty list.")
                docstring_tokens = []

            code=' '.join(code_tokens).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(docstring_tokens).replace('\n','')
            nl=' '.join(nl.strip().split())
            examples.append(
                Example(
                        idx = idx, # Sử dụng js['idx'] nếu nó tồn tại và hợp lệ, ngược lại dùng enumerate idx
                        source=code,
                        target = nl,
                        )
            )
    if not examples:
        logger.warning(f"No examples were successfully read from {filename}.")
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None") # For test stage, target is not used for generation input
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        
        # For MLM/CLM tasks, target_tokens are processed differently than seq2seq.
        # For seq2seq, target_ids are used as labels during training, and for teacher forcing.
        # During inference (test stage), we don't need target_ids in the same way for the input to decoder.
        # The model.generate or custom generation loop will handle decoder inputs.
        # However, for consistency with training and eval_ppl, we still create padded target_ids.
        
        # Common processing for target_ids (used as labels or for PPL calculation)
        processed_target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(processed_target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length_target = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length_target
        target_mask+=[0]*padding_length_target

        if example_index < 5 and stage=='train': # Log only for train and a few examples
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))

            logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

            logger.info("target_tokens (original): {}".format([x.replace('\u0120','_') for x in target_tokens]))
            logger.info("target_ids (for loss/ppl): {}".format(' '.join(map(str, target_ids))))
            logger.info("target_mask (for loss/ppl): {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids, # These are for calculating loss or PPL
                 source_mask,
                 target_mask, # Corresponding mask for target_ids
            )
        )
    return features


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.") # Sửa lại mô tả
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, # Giữ lại nếu có train
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, # Giữ lại nếu có train
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int, # Giữ lại nếu có train
                        help="Number of update steps between two evaluations.")
    parser.add_argument("--train_steps", default=-1, type=int, # Giữ lại nếu có train
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, # Giữ lại nếu có train
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda and torch.cuda.is_available() else 0
    else:  # Initializes the distributed backend
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if args.local_rank in [-1, 0]: # Chỉ master process tạo thư mục
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    #build model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, batch_first=True) # Thêm batch_first=True
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=12) # Sử dụng 12 hoặc config.num_hidden_layers
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("Reloading model from {}".format(args.load_model_path))
        device_to_load_on = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() else device
        logger.info(f"Attempting to load model to {device_to_load_on}.")
        try:
            model.load_state_dict(torch.load(args.load_model_path, map_location=device_to_load_on)) # Bỏ weights_only, sẽ thử cách này trước
        except RuntimeError as e:
            logger.warning(f"Failed to load model directly: {e}. This might be due to DataParallel/DDP wrapper or weights_only issues.")
            try:
                # Thử load với weights_only=True nếu file chỉ là weights
                logger.info("Retrying with weights_only=True.")
                model.load_state_dict(torch.load(args.load_model_path, map_location=device_to_load_on, weights_only=True))
            except Exception as e_weights_only:
                logger.error(f"Failed to load with weights_only=True: {e_weights_only}. Check model compatibility.")
                # Nếu model được lưu với DataParallel, cần gỡ wrapper
                try:
                    logger.info("Attempting to load by removing 'module.' prefix (common with DataParallel).")
                    state_dict = torch.load(args.load_model_path, map_location=device_to_load_on)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k # remove `module.`
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                except Exception as e_module:
                    logger.error(f"Failed to load by removing 'module.' prefix: {e_module}. Model loading failed.")
                    # Cân nhắc raise lỗi ở đây hoặc thoát nếu không load được model
        logger.info("Model loaded successfully.")

    model.to(device)

    if args.local_rank != -1:
        # Distributed training
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True) # find_unused_parameters có thể cần thiết
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        if not train_examples:
            logger.error(f"No training examples found in {args.train_filename}. Exiting.")
            return

        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

        num_train_optimization_steps =  args.train_steps if args.train_steps > 0 else int(len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs)
        if num_train_optimization_steps <= 0: # Thêm kiểm tra
            logger.error("Calculated num_train_optimization_steps is not positive. Check train_steps or num_train_epochs and data size.")
            return

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epochs = %f", args.num_train_epochs)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)


        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6
        # bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps, disable=args.local_rank not in [-1, 0])
        # train_dataloader=cycle(train_dataloader) # cycle có thể gây nhầm lẫn nếu không dùng đúng num_train_epochs
        eval_flag = True

        for epoch in range(int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{int(args.num_train_epochs)}", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch
                loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                current_loss = loss.item() # Lấy giá trị loss của batch hiện tại
                tr_loss += current_loss # Cộng dồn loss
                
                # Cập nhật bar description với loss trung bình của epoch hiện tại
                # train_loss=round(tr_loss/(nb_tr_steps+1),4) if nb_tr_steps > 0 else current_loss
                # bar.set_description("loss {}".format(train_loss))

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1 # Tăng nb_tr_steps sau khi xử lý batch
                
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    eval_flag = True
                    
                    # Log loss trung bình của các batch đã qua trong epoch này
                    avg_epoch_loss = tr_loss / (step + 1) 
                    bar.set_description(f"Epoch {epoch+1} Loss {avg_epoch_loss:.4f}")

                    if args.local_rank in [-1, 0] and args.do_eval and args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_flag:
                        #Eval model with dev dataset
                        # tr_loss_for_log = tr_loss # tr_loss đang là tổng loss của epoch, reset sau epoch
                        # nb_tr_steps_for_log = nb_tr_steps

                        eval_flag=False # Đánh dấu đã eval ở step này
                        if 'dev_loss' in dev_dataset:
                            eval_examples,eval_data=dev_dataset['dev_loss']
                        else:
                            logger.info("Reading dev examples for PPL...")
                            eval_examples = read_examples(args.dev_filename)
                            if not eval_examples:
                                logger.warning("No dev examples for PPL, skipping PPL evaluation.")
                                model.train() # Quay lại train mode
                                continue # Bỏ qua eval nếu không có data
                            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                            all_source_ids_dev = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                            all_source_mask_dev = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                            all_target_ids_dev = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                            all_target_mask_dev = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                            eval_data = TensorDataset(all_source_ids_dev,all_source_mask_dev,all_target_ids_dev,all_target_mask_dev)
                            dev_dataset['dev_loss']=eval_examples,eval_data

                        eval_sampler = SequentialSampler(eval_data)
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

                        logger.info("\n***** Running PPL evaluation *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", args.eval_batch_size)

                        model.eval()
                        eval_loss,tokens_num = 0,0
                        for batch_eval in eval_dataloader:
                            batch_eval = tuple(t.to(device) for t in batch_eval)
                            source_ids_eval,source_mask_eval,target_ids_eval,target_mask_eval = batch_eval
                            with torch.no_grad():
                                _,loss_eval,num = model(source_ids=source_ids_eval,source_mask=source_mask_eval,
                                                   target_ids=target_ids_eval,target_mask=target_mask_eval)
                            eval_loss += loss_eval.sum().item()
                            tokens_num += num.sum().item()

                        model.train() # Quay lại train mode
                        eval_loss_per_token = eval_loss / tokens_num if tokens_num > 0 else float('inf')
                        current_train_loss_avg = tr_loss / (step + 1) # Loss trung bình của epoch hiện tại

                        result = {'eval_ppl': round(np.exp(eval_loss_per_token),5) if tokens_num > 0 else float('inf'),
                                  'global_step': global_step,
                                  'train_loss': round(current_train_loss_avg,5)}
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                        logger.info("  "+"*"*20)

                        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)

                        if tokens_num > 0 and eval_loss_per_token < best_loss:
                            logger.info("  Best ppl: %s",round(np.exp(eval_loss_per_token),5))
                            best_loss=eval_loss_per_token
                            output_dir_best_ppl = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                            if not os.path.exists(output_dir_best_ppl):
                                os.makedirs(output_dir_best_ppl)
                            torch.save(model_to_save.state_dict(), os.path.join(output_dir_best_ppl, "pytorch_model.bin"))
                            logger.info(f"Saved best PPL model to {os.path.join(output_dir_best_ppl, 'pytorch_model.bin')}")

                        # Calculate bleu
                        logger.info("Reading dev examples for BLEU...")
                        if 'dev_bleu' in dev_dataset: # Chỉ lấy lại nếu chưa có hoặc muốn refresh
                             eval_examples_bleu, eval_data_bleu = dev_dataset['dev_bleu']
                        else:
                            eval_examples_bleu = read_examples(args.dev_filename)
                            if not eval_examples_bleu:
                                logger.warning("No dev examples for BLEU, skipping BLEU evaluation.")
                                model.train()
                                continue
                            # Sample một phần nhỏ để tính BLEU nhanh hơn trong quá trình train
                            sample_size = min(1000, len(eval_examples_bleu))
                            eval_examples_bleu = random.sample(eval_examples_bleu, sample_size)
                            logger.info(f"Sampled {sample_size} examples for BLEU evaluation.")

                            eval_features_bleu = convert_examples_to_features(eval_examples_bleu, tokenizer, args, stage='test') # stage='test' vì không dùng target_ids
                            all_source_ids_bleu = torch.tensor([f.source_ids for f in eval_features_bleu], dtype=torch.long)
                            all_source_mask_bleu = torch.tensor([f.source_mask for f in eval_features_bleu], dtype=torch.long)
                            eval_data_bleu = TensorDataset(all_source_ids_bleu,all_source_mask_bleu)
                            dev_dataset['dev_bleu'] = eval_examples_bleu, eval_data_bleu
                        
                        eval_sampler_bleu = SequentialSampler(eval_data_bleu)
                        eval_dataloader_bleu = DataLoader(eval_data_bleu, sampler=eval_sampler_bleu, batch_size=args.eval_batch_size, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

                        model.eval()
                        p_bleu=[]
                        for batch_bleu in tqdm(eval_dataloader_bleu, total=len(eval_dataloader_bleu), desc="Generating for BLEU (dev)"):
                            batch_bleu = tuple(t.to(device) for t in batch_bleu)
                            source_ids_b, source_mask_b = batch_bleu
                            with torch.no_grad():
                                preds_b = model(source_ids=source_ids_b, source_mask=source_mask_b)
                                for pred_candidates in preds_b:
                                    # Lấy beam tốt nhất
                                    best_hyp_ids = pred_candidates[0].cpu().numpy()
                                    best_hyp_ids = list(best_hyp_ids)
                                    if tokenizer.eos_token_id in best_hyp_ids:
                                        eos_idx = best_hyp_ids.index(tokenizer.eos_token_id)
                                        best_hyp_ids = best_hyp_ids[:eos_idx]
                                    elif 0 in best_hyp_ids and tokenizer.pad_token_id == 0:
                                        pad_idx = best_hyp_ids.index(0)
                                        best_hyp_ids = best_hyp_ids[:pad_idx]
                                    text = tokenizer.decode(best_hyp_ids,clean_up_tokenization_spaces=False, skip_special_tokens=True)
                                    p_bleu.append(text)
                        model.train() # Quay lại train mode

                        if len(p_bleu) == len(eval_examples_bleu):
                            predictions_for_bleu_eval = []
                            output_dev_output_path = os.path.join(args.output_dir,"dev.output")
                            output_dev_gold_path = os.path.join(args.output_dir,"dev.gold")
                            with open(output_dev_output_path,'w', encoding='utf-8') as f_out, \
                                 open(output_dev_gold_path,'w', encoding='utf-8') as f_gold:
                                for pred_text, gold_ex in zip(p_bleu, eval_examples_bleu):
                                    f_out.write(str(gold_ex.idx)+'\t'+pred_text.replace('\n',' ') +'\n')
                                    f_gold.write(str(gold_ex.idx)+'\t'+gold_ex.target.replace('\n',' ') +'\n')
                                    predictions_for_bleu_eval.append(str(gold_ex.idx)+'\t'+pred_text.replace('\n',' '))

                            if predictions_for_bleu_eval:
                                (goldMap, predictionMap) = bleu.computeMaps(predictions_for_bleu_eval, output_dev_gold_path)
                                if goldMap and predictionMap:
                                    dev_bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                                    logger.info("  %s (dev) = %s "%("bleu-4",str(dev_bleu_score)))
                                    if dev_bleu_score > best_bleu:
                                        logger.info("  Best bleu (dev):%s",dev_bleu_score)
                                        best_bleu=dev_bleu_score
                                        output_dir_best_bleu = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                                        if not os.path.exists(output_dir_best_bleu):
                                            os.makedirs(output_dir_best_bleu)
                                        model_to_save_bleu = model.module if hasattr(model, 'module') else model
                                        torch.save(model_to_save_bleu.state_dict(), os.path.join(output_dir_best_bleu, "pytorch_model.bin"))
                                        logger.info(f"Saved best BLEU model to {os.path.join(output_dir_best_bleu, 'pytorch_model.bin')}")
                                else:
                                    logger.warning("Could not compute BLEU maps for dev set.")
                            else:
                                logger.warning("No predictions formatted for BLEU eval on dev set.")
                        else:
                            logger.warning(f"Mismatch in dev BLEU predictions ({len(p_bleu)}) and examples ({len(eval_examples_bleu)}). Skipping BLEU.")
                        logger.info("  "+"*"*20)
            # Kết thúc vòng lặp batch
            logger.info(f"Finished Epoch {epoch+1}. Average training loss for epoch: {tr_loss/len(train_dataloader):.4f}")
            tr_loss = 0 # Reset tr_loss cho epoch tiếp theo
            nb_tr_steps = 0 # Reset nb_tr_steps cho epoch tiếp theo
        # Kết thúc vòng lặp epoch
        logger.info("***** Training finished *****")


    if args.do_test:
        files_to_test=[]
        if args.dev_filename is not None and os.path.exists(args.dev_filename): # Kiểm tra file tồn tại
            files_to_test.append(args.dev_filename)
            logger.info(f"Added dev file for testing: {args.dev_filename}")
        elif args.dev_filename:
            logger.warning(f"Dev file {args.dev_filename} not found, will not be tested.")

        if args.test_filename is not None and os.path.exists(args.test_filename): # Kiểm tra file tồn tại
            files_to_test.append(args.test_filename)
            logger.info(f"Added test file for testing: {args.test_filename}")
        elif args.test_filename:
            logger.warning(f"Test file {args.test_filename} not found, will not be tested.")


        if not files_to_test:
            logger.warning("No valid files specified for --do_test. Exiting test phase.")
            return

        for file_idx, current_file_to_test in enumerate(files_to_test):
            logger.info("="*50)
            file_basename = os.path.basename(current_file_to_test)
            logger.info(f"Processing file (idx={file_idx}, name={file_basename}): {current_file_to_test}")

            eval_examples = read_examples(current_file_to_test)
            if not eval_examples:
                logger.error(f"No examples read from file: {current_file_to_test}. Skipping evaluation for this file.")
                continue

            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids,all_source_mask)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

            model.eval()
            p_test = [] # Predictions for the current file
            logger.info(f"Starting prediction generation for {file_basename}...")
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader), desc=f"Predicting for {file_basename}"):
                batch = tuple(t.to(device) for t in batch)
                source_ids_test, source_mask_test = batch
                with torch.no_grad():
                    preds_test_batch = model(source_ids=source_ids_test, source_mask=source_mask_test)
                    for single_pred_candidates in preds_test_batch:
                        best_hyp_ids = single_pred_candidates[0].cpu().numpy()
                        best_hyp_ids = list(best_hyp_ids)
                        # Xử lý cắt chuỗi tại EOS hoặc PAD
                        processed_ids = []
                        for token_id in best_hyp_ids:
                            if token_id == tokenizer.eos_token_id or \
                               (token_id == tokenizer.pad_token_id and tokenizer.pad_token_id != tokenizer.eos_token_id): # Tránh cắt nếu pad trùng eos và eos chưa xuất hiện
                                break
                            processed_ids.append(token_id)
                        
                        text = tokenizer.decode(processed_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                        p_test.append(text)
            logger.info(f"Finished prediction generation for {file_basename}. Generated {len(p_test)} predictions.")

            if len(p_test) != len(eval_examples):
                logger.error(f"Mismatch in number of predictions ({len(p_test)}) and examples ({len(eval_examples)}) for file {current_file_to_test}. Skipping EM and BLEU.")
                continue

            # --- Tính EM và ghi file ---
            exact_matches_test = []
            predictions_for_bleu_test_file = []

            output_file_path_test = os.path.join(args.output_dir, f"test_{file_idx}_{file_basename}.output")
            gold_file_path_test = os.path.join(args.output_dir, f"test_{file_idx}_{file_basename}.gold")

            logger.info(f"Writing predictions to: {output_file_path_test}")
            logger.info(f"Writing gold references to: {gold_file_path_test}")

            with open(output_file_path_test, 'w', encoding='utf-8') as f_out, \
                 open(gold_file_path_test, 'w', encoding='utf-8') as f_gold:
                for i, (pred_text, gold_example) in enumerate(zip(p_test, eval_examples)):
                    normalized_pred = pred_text.strip().lower() # Chuẩn hóa cho EM
                    normalized_gold = gold_example.target.strip().lower() # Chuẩn hóa cho EM

                    is_em = (normalized_pred == normalized_gold)
                    exact_matches_test.append(is_em)

                    f_out.write(str(gold_example.idx) + '\t' + pred_text.replace('\n', ' ') + '\n')
                    f_gold.write(str(gold_example.idx) + '\t' + gold_example.target.replace('\n', ' ') + '\n')
                    predictions_for_bleu_test_file.append(str(gold_example.idx) + '\t' + pred_text.replace('\n', ' '))

            logger.info(f"Finished writing files for {file_basename}.")

            # Tính và in EM score
            if exact_matches_test:
                em_score_test = np.mean(exact_matches_test) * 100
                num_em_correct_test = sum(exact_matches_test)
                logger.info(f"  Exact Match (EM) for {file_basename} = {em_score_test:.2f}% ({num_em_correct_test}/{len(exact_matches_test)})")
            else:
                logger.info(f"  Exact Match (EM) for {file_basename} = N/A (no predictions to compare)")
            logger.info("  " + "*" * 20)

            # --- Tính BLEU ---
            if not predictions_for_bleu_test_file:
                logger.error(f"No predictions formatted for BLEU for file {file_basename}. Skipping BLEU.")
                continue

            logger.info(f"Calculating BLEU for {file_basename} using {len(predictions_for_bleu_test_file)} predictions and gold file {gold_file_path_test}")
            try:
                if not os.path.exists(gold_file_path_test) or os.path.getsize(gold_file_path_test) == 0:
                    logger.error(f"Gold file {gold_file_path_test} is missing or empty. Cannot compute BLEU.")
                    continue

                (goldMap, predictionMap) = bleu.computeMaps(predictions_for_bleu_test_file, gold_file_path_test)

                if goldMap is None or predictionMap is None or not goldMap or not predictionMap:
                     logger.error(f"computeMaps did not return valid maps for {file_basename}. Skipping BLEU.")
                     continue

                bleu_score_results_test = bleu.bleuFromMaps(goldMap, predictionMap)
                if bleu_score_results_test and isinstance(bleu_score_results_test, (list, tuple)) and len(bleu_score_results_test) > 0:
                    current_bleu_score = round(bleu_score_results_test[0], 2)
                    logger.info("  %s for %s = %s " % ("bleu-4", file_basename, str(current_bleu_score)))
                else:
                    logger.error(f"bleuFromMaps did not return a valid BLEU score for {file_basename}. Result: {bleu_score_results_test}")
                logger.info("  " + "*" * 20)

            except Exception as e:
                logger.error(f"ERROR calculating BLEU for file {file_basename}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            logger.info("="*50)
        # Kết thúc vòng lặp qua các file test
        logger.info("***** Testing finished *****")

if __name__ == "__main__":
    main()


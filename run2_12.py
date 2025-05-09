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
import os # <<-- THÊM IMPORT NÀY
import sys
import bleu # Đảm bảo module bleu này tồn tại và hoạt động đúng
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
from model import Seq2Seq # Đảm bảo file model.py tồn tại và có class Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
# from torch.optim import Adam
from torch.optim import AdamW # Sử dụng AdamW từ transformers thường tốt hơn
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
    try: # Thêm try-except để bắt lỗi đọc file
        with open(filename,encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try: # Thêm try-except cho mỗi dòng JSON
                    line=line.strip()
                    js=json.loads(line)
                    if 'idx' not in js:
                        js['idx']=idx
                    # Kiểm tra key tồn tại trước khi truy cập
                    code_tokens = js.get('code_tokens', [])
                    docstring_tokens = js.get('docstring_tokens', [])

                    code=' '.join(code_tokens).replace('\n',' ')
                    code=' '.join(code.strip().split())
                    nl=' '.join(docstring_tokens).replace('\n','')
                    nl=' '.join(nl.strip().split())
                    examples.append(
                        Example(
                                idx = js['idx'], # Sử dụng idx từ json nếu có
                                source=code,
                                target = nl,
                                )
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {idx+1} in {filename}: {line[:100]}...")
                except KeyError as e:
                     logger.warning(f"Skipping line {idx+1} due to missing key {e} in {filename}: {line[:100]}...")
    except FileNotFoundError:
         logger.error(f"Error: File not found at {filename}")
         return [] # Trả về danh sách rỗng nếu file không tồn tại

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
             # Nên có target rỗng hoặc một token đặc biệt thay vì "None"
             # target_tokens = tokenizer.tokenize("")[:args.max_target_length-2]
            target_tokens = [] # Target rỗng cho test
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]

        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length

        # Bỏ bớt log không cần thiết trong vòng lặp này để tránh quá nhiều output
        # if example_index < 5:
        #     if stage=='train':
        #         # ... (logging) ...

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
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
                        help="Whether to run eval on the test set.") # Sửa comment
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
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
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
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
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
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=12) # Số layer decoder có thể cần điều chỉnh
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        try: # Thêm try-except để bắt lỗi load model
            model.load_state_dict(torch.load(args.load_model_path, map_location=args.device))
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {args.load_model_path}. Exiting.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model state dict from {args.load_model_path}: {e}", exc_info=True)
            # Có thể bạn muốn thoát ở đây hoặc tiếp tục với model chưa train
            # sys.exit(1)

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    # --- Phần Training (Giữ nguyên logic gốc) ---
    if args.do_train:
        # ... (code training giữ nguyên như file gốc của bạn) ...
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
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
        # Đảm bảo batch size hợp lệ
        train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        if train_batch_size == 0:
            logger.error("Effective batch size is zero. Check train_batch_size and gradient_accumulation_steps.")
            train_batch_size = 1 # Đặt giá trị tối thiểu để tránh lỗi chia cho 0
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)


        # Xác định num_train_optimization_steps
        if args.max_steps > 0:
             num_train_optimization_steps = args.max_steps
             args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
             num_train_optimization_steps = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
             if args.train_steps > 0: # Ưu tiên train_steps nếu được cung cấp
                 num_train_optimization_steps = args.train_steps


        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)


        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num Epochs = %d", int(args.num_train_epochs)) # Sử dụng num_train_epochs đã tính toán lại nếu cần
        logger.info("  Instantaneous batch size per GPU = %d", train_batch_size) # Batch size thực tế mỗi step
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)


        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6
        # Sửa lại logic vòng lặp để dựa trên epoch hoặc steps
        if args.max_steps <= 0: # Lặp theo epoch
            train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
            global_step = 0 # Reset global_step nếu lặp theo epoch
        else: # Lặp theo step
             train_iterator = trange(num_train_optimization_steps, desc="Step", disable=args.local_rank not in [-1, 0])
             # global_step được tính trong vòng lặp step

        train_dataloader_iter=cycle(train_dataloader)
        eval_flag = True

        for epoch_or_step in train_iterator: # Lặp qua epoch hoặc step tùy cấu hình
            if args.max_steps <=0: # Nếu lặp theo epoch, cần lặp qua dataloader bên trong
                 epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
                 for step, batch in enumerate(epoch_iterator):
                      model.train() # Đảm bảo model ở chế độ train
                      batch = tuple(t.to(device) for t in batch)
                      source_ids,source_mask,target_ids,target_mask = batch
                      loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)

                      if args.n_gpu > 1:
                          loss = loss.mean()
                      if args.gradient_accumulation_steps > 1:
                          loss = loss / args.gradient_accumulation_steps

                      loss.backward()
                      tr_loss += loss.item()

                      if (step + 1) % args.gradient_accumulation_steps == 0:
                          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # Thêm clip grad norm
                          optimizer.step()
                          scheduler.step()
                          optimizer.zero_grad()
                          global_step += 1
                          eval_flag = True

                          # Log loss trung bình cho epoch hiện tại
                          train_loss_avg = tr_loss / global_step if global_step > 0 else 0
                          epoch_iterator.set_description(f"Epoch {epoch_or_step} Iteration Loss: {loss.item():.4f}, Avg Loss: {train_loss_avg:.4f}")

                          # --- Evaluation Logic ---
                          if args.do_eval and args.local_rank in [-1, 0] and (global_step % args.eval_steps == 0) and eval_flag:
                              # ... (phần eval giữ nguyên logic như file gốc của bạn, nhưng thực hiện trong rank 0) ...
                                # Eval model with dev dataset
                              tr_loss_val = 0 # Đặt lại tr_loss cho eval
                              nb_tr_examples_val, nb_tr_steps_val = 0, 0
                              eval_flag=False
                              if 'dev_loss' in dev_dataset:
                                  eval_examples,eval_data=dev_dataset['dev_loss']
                              else:
                                  eval_examples = read_examples(args.dev_filename)
                                  eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                                  all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                                  all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                                  all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                                  all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                                  eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
                                  dev_dataset['dev_loss']=eval_examples,eval_data
                              eval_sampler = SequentialSampler(eval_data)
                              eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                              logger.info("\n***** Running evaluation *****")
                              logger.info("  Num examples = %d", len(eval_examples))
                              logger.info("  Batch size = %d", args.eval_batch_size)

                              #Start Evaling model
                              model.eval()
                              eval_loss,tokens_num = 0,0
                              for batch_eval in eval_dataloader: # Đổi tên biến batch
                                  batch_eval = tuple(t.to(device) for t in batch_eval)
                                  source_ids_eval,source_mask_eval,target_ids_eval,target_mask_eval = batch_eval

                                  with torch.no_grad():
                                      _,loss_eval,num = model(source_ids=source_ids_eval,source_mask=source_mask_eval,
                                                         target_ids=target_ids_eval,target_mask=target_mask_eval)
                                  if args.n_gpu > 1:
                                      loss_eval = loss_eval.mean() # Xử lý multi-gpu loss
                                  eval_loss += loss_eval.item() * num.sum().item() # Tính tổng loss theo token
                                  tokens_num += num.sum().item()
                              #Pring loss of dev dataset
                              model.train()
                              eval_loss = eval_loss / tokens_num if tokens_num > 0 else 0 # Tránh chia cho 0
                              result = {'eval_ppl': round(np.exp(eval_loss),5) if eval_loss > -float('inf') else float('inf'), # Handle potential -inf
                                        'global_step': global_step,
                                        'train_loss': round(train_loss_avg,5)} # Sử dụng train_loss_avg
                              for key in sorted(result.keys()):
                                  logger.info("  %s = %s", key, str(result[key]))
                              logger.info("  "+"*"*20)

                              #save last checkpoint
                              last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                              if not os.path.exists(last_output_dir):
                                  os.makedirs(last_output_dir)
                              model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                              output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                              torch.save(model_to_save.state_dict(), output_model_file)
                              if eval_loss<best_loss:
                                  logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                                  logger.info("  "+"*"*20)
                                  best_loss=eval_loss
                                  # Save best checkpoint for best ppl
                                  output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                                  if not os.path.exists(output_dir):
                                      os.makedirs(output_dir)
                                  model_to_save = model.module if hasattr(model, 'module') else model
                                  output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                  torch.save(model_to_save.state_dict(), output_model_file)


                              #Calculate bleu
                              if 'dev_bleu' in dev_dataset:
                                  eval_examples_bleu,eval_data_bleu=dev_dataset['dev_bleu'] # Đổi tên biến
                              else:
                                  eval_examples_bleu = read_examples(args.dev_filename) # Đổi tên biến
                                  eval_examples_bleu = random.sample(eval_examples_bleu,min(1000,len(eval_examples_bleu))) # Giảm kích thước sample để nhanh hơn
                                  eval_features_bleu = convert_examples_to_features(eval_examples_bleu, tokenizer, args,stage='test') # Đổi tên biến
                                  all_source_ids_bleu = torch.tensor([f.source_ids for f in eval_features_bleu], dtype=torch.long)
                                  all_source_mask_bleu = torch.tensor([f.source_mask for f in eval_features_bleu], dtype=torch.long)
                                  eval_data_bleu = TensorDataset(all_source_ids_bleu,all_source_mask_bleu) # Đổi tên biến
                                  dev_dataset['dev_bleu']=eval_examples_bleu,eval_data_bleu


                              eval_sampler_bleu = SequentialSampler(eval_data_bleu) # Đổi tên biến
                              eval_dataloader_bleu = DataLoader(eval_data_bleu, sampler=eval_sampler_bleu, batch_size=args.eval_batch_size) # Đổi tên biến

                              model.eval()
                              p=[]
                              for batch_bleu in eval_dataloader_bleu: # Đổi tên biến
                                  batch_bleu = tuple(t.to(device) for t in batch_bleu)
                                  source_ids_bleu,source_mask_bleu= batch_bleu
                                  with torch.no_grad():
                                      preds = model(source_ids=source_ids_bleu,source_mask=source_mask_bleu)
                                      for pred in preds:
                                          # Lấy beam đầu tiên (tốt nhất)
                                          first_beam_output = pred[0]
                                          t = first_beam_output.cpu().numpy()
                                          t = list(t)
                                          # Sử dụng eos_token_id của tokenizer thay vì số 0 cứng
                                          if tokenizer.eos_token_id in t:
                                             t = t[:t.index(tokenizer.eos_token_id)]
                                          # Sử dụng bos_token_id của tokenizer thay vì cls_token_id
                                          if tokenizer.bos_token_id in t:
                                               t = t[t.index(tokenizer.bos_token_id)+1:]
                                          text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                                          p.append(text)
                              model.train()
                              predictions=[]
                              dev_gold_path = os.path.join(args.output_dir, "dev.gold")
                              dev_output_path = os.path.join(args.output_dir, "dev.output")

                              with open(dev_output_path,'w', encoding='utf-8') as f, \
                                   open(dev_gold_path,'w', encoding='utf-8') as f1:
                                  for ref,gold in zip(p, eval_examples_bleu): # Dùng p (best candidate) để ghi file output
                                      predictions.append(str(gold.idx)+'\t'+ref) # Dùng cho computeMaps
                                      f.write(str(gold.idx)+'\t'+ref+'\n')
                                      f1.write(str(gold.idx)+'\t'+gold.target+'\n')

                              try:
                                  (goldMap, predictionMap) = bleu.computeMaps(predictions, dev_gold_path)
                                  dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                                  logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                              except Exception as e:
                                  logger.error(f"Error calculating BLEU score: {e}", exc_info=True)
                                  dev_bleu = 0.0 # Đặt giá trị mặc định

                              logger.info("  "+"*"*20)
                              if dev_bleu>best_bleu:
                                  logger.info("  Best bleu:%s",dev_bleu)
                                  logger.info("  "+"*"*20)
                                  best_bleu=dev_bleu
                                  # Save best checkpoint for best bleu
                                  output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                                  if not os.path.exists(output_dir):
                                      os.makedirs(output_dir)
                                  model_to_save = model.module if hasattr(model, 'module') else model
                                  output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                  torch.save(model_to_save.state_dict(), output_model_file)
                          # --- Kết thúc Evaluation Logic ---

                      if args.max_steps > 0 and global_step >= args.max_steps:
                          break # Thoát vòng lặp step bên trong nếu đạt max_steps
                 if args.max_steps > 0 and global_step >= args.max_steps:
                     break # Thoát vòng lặp epoch bên ngoài nếu đạt max_steps

            # --- Logic cũ cho lặp theo steps (cần xem lại nếu bạn dùng max_steps) ---
            # else: # Lặp theo step
            #    # ... (logic tương tự như lặp theo epoch nhưng không có vòng lặp epoch_iterator) ...
            #    # Lấy batch trực tiếp
            #     batch = next(train_dataloader_iter)
            #     model.train()
            #     batch = tuple(t.to(device) for t in batch)
            #     source_ids,source_mask,target_ids,target_mask = batch
            #     # ... (tính loss, backward, ...)
            #     # ... (update optimizer, scheduler, global_step)
            #     # ... (evaluation logic giống như trên khi global_step % args.eval_steps == 0)


    # --- Phần Test (Thêm Debugging và sửa lỗi tính EM) ---
    if args.do_test:
        files=[]
        # Chỉ thêm file nếu nó được cung cấp và tồn tại
        if args.dev_filename is not None and os.path.exists(args.dev_filename):
            files.append(args.dev_filename)
        if args.test_filename is not None and os.path.exists(args.test_filename):
            files.append(args.test_filename)
        else:
             logger.warning(f"Test filename {args.test_filename} not provided or does not exist.")


        for file_idx, file in enumerate(files): # Đổi tên biến idx thành file_idx
            logger.info("***** Running testing on %s *****", file)
            eval_examples = read_examples(file)
            if not eval_examples: # Kiểm tra nếu đọc file bị lỗi
                logger.error(f"Could not read examples from {file}. Skipping testing.")
                continue

            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids,all_source_mask) # Target không cần thiết cho input test

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = [] # Danh sách dự đoán tốt nhất (best candidate)
            print(f"DEBUG: Starting evaluation loop for file: {file}", flush=True)
            # Vòng lặp tạo dự đoán
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval bleu for test set {file_idx}"):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    # Giả sử model() trả về danh sách các beam cho mỗi input
                    # preds có thể là list[list[tensor]] hoặc cấu trúc khác tùy model.Seq2Seq
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    if not isinstance(preds, list): # Xử lý nếu preds không phải list
                         logger.warning("Model output 'preds' is not a list. Attempting to handle.")
                         # Cố gắng chuyển đổi hoặc xử lý tùy theo cấu trúc thực tế
                         if isinstance(preds, torch.Tensor):
                             preds = preds.tolist() # Ví dụ nếu là tensor
                         else:
                             preds = [] # Bỏ qua batch này nếu không xử lý được

                    for pred_beams in preds: # pred_beams là list các beam cho một input
                        best_candidate_text = ""
                        # Kiểm tra pred_beams là list và không rỗng
                        if isinstance(pred_beams, list) and pred_beams:
                             # Giả sử beam đầu tiên là tốt nhất
                            first_beam_output = pred_beams[0]
                            # Kiểm tra first_beam_output là tensor
                            if isinstance(first_beam_output, torch.Tensor):
                                t = first_beam_output.cpu().numpy()
                                t = list(t)
                                # Chuẩn hóa việc cắt token EOS
                                if tokenizer.eos_token_id in t:
                                    t = t[:t.index(tokenizer.eos_token_id)]
                                # Bỏ token BOS nếu có (một số model thêm vào đầu)
                                if t and t[0] == tokenizer.bos_token_id:
                                     t = t[1:]
                                # Bỏ token CLS nếu có (một số model thêm vào đầu)
                                elif t and t[0] == tokenizer.cls_token_id:
                                     t = t[1:]
                                best_candidate_text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            else:
                                logger.warning(f"Unexpected type for beam output: {type(first_beam_output)}")
                        else:
                             logger.warning(f"Unexpected structure or empty beam list for a prediction: {pred_beams}")

                        p.append(best_candidate_text) # Thêm dự đoán tốt nhất

            print(f"DEBUG: Finished evaluation loop. Generated predictions: {len(p)}", flush=True)

            # --- Ghi File và Tính Metrics ---
            if len(p) != len(eval_examples):
                 logger.error(f"CRITICAL ERROR: Mismatch between number of predictions ({len(p)}) and examples ({len(eval_examples)}) for file {file}. Skipping metrics calculation.")
                 continue # Bỏ qua file này nếu số lượng không khớp

            # Ghi file output (chứa best candidate) và file gold
            # Sử dụng tên file duy nhất cho mỗi tập test
            file_basename = os.path.basename(file).replace('.jsonl', '').replace('.json', '')
            output_file_path = os.path.join(args.output_dir, f"{file_basename}.output")
            gold_file_path = os.path.join(args.output_dir, f"{file_basename}.gold")

            print(f"DEBUG: Writing output to {output_file_path} and gold to {gold_file_path}...", flush=True)
            predictions_for_bleu_compute = [] # Cần format "idx\tpred" cho bleu.computeMaps
            try:
                with open(output_file_path,'w', encoding='utf-8') as f_out, \
                     open(gold_file_path,'w', encoding='utf-8') as f_gold:
                    for pred_text, gold_example in zip(p, eval_examples):
                        # Ghi best candidate vào file output
                        f_out.write(str(gold_example.idx) + '\t' + pred_text + '\n')
                        f_gold.write(str(gold_example.idx) + '\t' + gold_example.target + '\n')
                        predictions_for_bleu_compute.append(str(gold_example.idx) + '\t' + pred_text)
                print("DEBUG: Finished writing files.", flush=True)
            except IOError as e:
                 print(f"ERROR writing output/gold files: {e}", flush=True)
                 logger.error(f"Error writing files for {file}: {e}", exc_info=True)
                 continue # Bỏ qua tính toán metrics nếu ghi file lỗi


            # Tính BLEU
            print("DEBUG: Computing BLEU...", flush=True)
            dev_bleu = 0.0 # Giá trị mặc định
            try:
                # Đảm bảo file gold tồn tại trước khi tính
                if os.path.exists(gold_file_path) and predictions_for_bleu_compute:
                    # Sử dụng predictions_for_bleu_compute đã tạo
                    (goldMap, predictionMap) = bleu.computeMaps(predictions_for_bleu_compute, gold_file_path)
                    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                    print(f"DEBUG: BLEU computed: {dev_bleu}", flush=True)
                elif not os.path.exists(gold_file_path):
                     logger.error(f"Gold file {gold_file_path} not found for BLEU calculation.")
                else: # predictions_for_bleu_compute rỗng
                     logger.error(f"Prediction list is empty for BLEU calculation.")

            except Exception as e:
                print(f"ERROR during BLEU computation: {e}", flush=True)
                logger.error("ERROR during BLEU computation:", exc_info=True)
                dev_bleu = 0.0 # Gán giá trị mặc định

            # Tính EM
            print("DEBUG: Computing EM...", flush=True)
            match_count = 0
            for pred_text, gold_example in zip(p, eval_examples):
                 pred_clean = pred_text.strip()
                 gold_clean = gold_example.target.strip()
                 if pred_clean == gold_clean:
                     match_count += 1
            exact_match_score = (match_count / len(eval_examples)) * 100 if len(eval_examples) > 0 else 0.0
            print(f"DEBUG: EM computed: {exact_match_score:.2f}", flush=True)

            # --- In kết quả cuối cùng ---
            print("DEBUG: Logging final results...", flush=True)
            logger.info("***** Results for %s *****", file) # Log tên file
            logger.info("  %s = %s", "BLEU-4", str(dev_bleu))
            logger.info("  %s = %.2f%%", "EM", exact_match_score)
            logger.info("  "+"*"*20)
            print("DEBUG: Finished logging final results.", flush=True)
            # --- Kết thúc In kết quả ---


if __name__ == "__main__":
    main()

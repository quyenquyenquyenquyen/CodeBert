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
from torch.optim import AdamW # Sử dụng AdamW từ transformers
import glob # Đã có
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
    try:
        with open(filename,encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    line=line.strip()
                    js=json.loads(line)
                    # Đảm bảo 'idx' luôn tồn tại
                    current_idx = js.get('idx', idx) # Ưu tiên idx trong file, nếu không có thì dùng line number

                    code_tokens = js.get('code_tokens', [])
                    docstring_tokens = js.get('docstring_tokens', [])

                    code=' '.join(code_tokens).replace('\n',' ')
                    code=' '.join(code.strip().split())
                    nl=' '.join(docstring_tokens).replace('\n','')
                    nl=' '.join(nl.strip().split())
                    examples.append(
                        Example(
                                idx = current_idx, # Sử dụng idx đã xác định
                                source=code,
                                target = nl,
                                )
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {idx+1} in {filename}: {line[:100]}...")
                except KeyError as e:
                     logger.warning(f"Skipping line {idx+1} due to missing key {e} in {filename}: {line[:100]}...")
    except FileNotFoundError:
         logger.error(f"Error: File not found at {filename}. Returning empty list.")
         return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {filename}: {e}", exc_info=True)
        return []
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
            target_tokens = [] # For test stage, target is not needed for model input
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]

        # Add <s> and </s> tokens
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length

        if example_index < 0: # Tắt log này để tránh spam console
            if stage=='train':
                logger.info("*** Example ***")
                # ... (logging) ...

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
                        help="Whether to run eval on the test set.") # Sửa lại comment
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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, PyTorch Cuda Available: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), torch.cuda.is_available())
    args.device = device # Đảm bảo args.device được gán đúng
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    #build model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=12) # Có thể 6 layer phù hợp hơn nếu encoder là 12 layer
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("Reloading model from {}".format(args.load_model_path))
        print(f"DEBUG: Attempting to load model from {args.load_model_path} onto device {args.device}", flush=True)
        try:
            # SỬA Ở ĐÂY: Thêm map_location=args.device
            # Điều này sẽ đảm bảo model được tải đúng cách dù nó được lưu trên CPU hay GPU,
            # và môi trường hiện tại có GPU hay không.
            loaded_state_dict = torch.load(args.load_model_path, map_location=args.device)
            model.load_state_dict(loaded_state_dict)
            logger.info("Model reloaded successfully from {} using map_location.".format(args.load_model_path))
            print(f"DEBUG: Model successfully loaded from {args.load_model_path}", flush=True)
        except FileNotFoundError:
            logger.error(f"Model file not found at {args.load_model_path}. Exiting.")
            sys.exit(1)
        except RuntimeError as e:
            logger.error(f"RuntimeError loading model state dict from {args.load_model_path}: {e}", exc_info=True)
            logger.error("This might be due to a mismatch in model architecture or the checkpoint being saved on a different device type without proper mapping during load.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading model from {args.load_model_path}: {e}", exc_info=True)
            sys.exit(1)


    model.to(args.device) # Đảm bảo model được chuyển lên device sau khi tải (nếu map_location là cpu)
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


    # --- PHẦN TRAINING ---
    if args.do_train:
        # ... (Logic training giữ nguyên như file gốc của bạn, chỉ sửa nhỏ) ...
        # (Bao gồm cả logic resume của bạn nếu có, hoặc logic tính num_train_optimization_steps)
        if args.load_model_path and 'checkpoint-step' in args.load_model_path and os.path.exists(args.load_model_path):
            logger.info(f"Training is resuming/continuing from a loaded model: {args.load_model_path}")
            # Logic resume step của bạn (đã có)
            import re
            m = re.search(r'checkpoint-step-(\d+)', args.load_model_path)
            resume_step = 0
            if m:
                resume_step = int(m.group(1))
            global_step = resume_step
            nb_tr_steps = resume_step
        else: # Bắt đầu training mới
            global_step = 0
            nb_tr_steps = 0


        train_examples = read_examples(args.train_filename)
        if not train_examples:
            logger.error(f"No training examples found in {args.train_filename}. Exiting.")
            sys.exit(1)

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

        actual_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        if actual_train_batch_size == 0: # Tránh lỗi chia cho 0
            logger.warning("Effective batch size is 0. Setting to 1.")
            actual_train_batch_size = 1

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=actual_train_batch_size, drop_last=True)


        if args.max_steps > 0:
            num_train_optimization_steps = args.max_steps
            # args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1 # Không cần thiết nếu dùng max_steps
        elif args.train_steps > 0 : # Ưu tiên train_steps nếu được cung cấp
            num_train_optimization_steps = args.train_steps
        else: # Tính theo num_train_epochs
            num_train_optimization_steps = int(len(train_dataloader) * args.num_train_epochs)


        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Instantaneous batch size per GPU = %d", actual_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)
        logger.info("  Starting from global step = %d", global_step)


        model.train()
        dev_dataset={}
        tr_loss=0.0 # Reset tr_loss
        best_bleu, best_loss = 0, 1e6 # Khởi tạo lại best metrics

        # Khởi tạo bar với initial step
        bar = tqdm(range(global_step, num_train_optimization_steps), total=num_train_optimization_steps, initial=global_step, desc="Training")
        train_dataloader_iter = cycle(train_dataloader) # Đổi tên biến để tránh nhầm lẫn
        eval_flag = True

        for current_step in bar: # Lặp qua các step còn lại
            batch = next(train_dataloader_iter)
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            tr_loss += loss.item()
            
            # Cập nhật nb_tr_steps ở đây, sau khi backward
            # nb_tr_steps += 1 # nb_tr_steps sẽ tương đương current_step + 1 - global_step (nếu global_step ban đầu > 0)
                            # Hoặc đơn giản là theo dõi số lần backward đã thực hiện từ khi resume
            
            if (current_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1 # Chỉ tăng global_step khi optimizer step
                eval_flag = True
                
                # Hiển thị loss trung bình kể từ lần eval cuối hoặc từ đầu
                # Chia cho số lần optimizer đã step kể từ khi tr_loss được reset (sau eval)
                # Hoặc một cách đơn giản hơn, hiển thị loss của step hiện tại
                avg_loss_display = tr_loss / (global_step - resume_step +1) if (global_step - resume_step +1) > 0 else loss.item()
                bar.set_description("Step {}/{} loss {:.4f}".format(global_step, num_train_optimization_steps, avg_loss_display))


                if args.do_eval and args.local_rank in [-1, 0] and (global_step % args.eval_steps == 0) and eval_flag:
                    eval_flag=False
                    # --- EVALUATION LOGIC (giữ nguyên từ code gốc của bạn, nhưng trong rank 0) ---
                    if 'dev_loss' in dev_dataset:
                        eval_examples,eval_data=dev_dataset['dev_loss']
                    else:
                        eval_examples = read_examples(args.dev_filename)
                        if not eval_examples: logger.warning(f"No dev examples for PPL eval from {args.dev_filename}"); continue
                        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                        all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                        all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                        eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
                        dev_dataset['dev_loss']=eval_examples,eval_data

                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    logger.info("\n***** Running PPL evaluation on Dev Set *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    model.eval()
                    eval_loss, tokens_num = 0,0
                    for batch_eval_ppl in eval_dataloader:
                        batch_eval_ppl = tuple(t.to(device) for t in batch_eval_ppl)
                        source_ids_e,source_mask_e,target_ids_e,target_mask_e = batch_eval_ppl
                        with torch.no_grad():
                            _,loss_e,num = model(source_ids=source_ids_e,source_mask=source_mask_e,
                                               target_ids=target_ids_e,target_mask=target_mask_e)
                        if args.n_gpu > 1: loss_e = loss_e.mean()
                        eval_loss += loss_e.item() * num.sum().item() # Tổng loss theo token
                        tokens_num += num.sum().item()

                    model.train()
                    eval_loss = eval_loss / tokens_num if tokens_num > 0 else 0
                    current_ppl = round(np.exp(eval_loss),5) if eval_loss > -float('inf') else float('inf')
                    result = {'eval_ppl': current_ppl,
                              'global_step': global_step,
                              'train_loss': round(avg_loss_display,5)} # Sử dụng loss trung bình gần nhất
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                    logger.info("  "+"*"*20)

                    if current_ppl < best_loss: # best_loss là best_ppl
                        logger.info("  Best ppl improved from %s to %s", best_loss, current_ppl)
                        best_loss = current_ppl
                        output_dir_best_ppl = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                        if not os.path.exists(output_dir_best_ppl): os.makedirs(output_dir_best_ppl)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir_best_ppl, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info(f"Saved best PPL model to {output_model_file}")
                    
                    tr_loss = 0.0 # Reset tr_loss sau mỗi lần eval

                    # Calculate bleu
                    if 'dev_bleu' in dev_dataset:
                        eval_examples_b,eval_data_b=dev_dataset['dev_bleu']
                    else:
                        eval_examples_b = read_examples(args.dev_filename)
                        if not eval_examples_b: logger.warning(f"No dev examples for BLEU eval from {args.dev_filename}"); continue
                        eval_examples_b = random.sample(eval_examples_b,min(1000,len(eval_examples_b)))
                        eval_features_b = convert_examples_to_features(eval_examples_b, tokenizer, args,stage='test')
                        all_source_ids_b = torch.tensor([f.source_ids for f in eval_features_b], dtype=torch.long)
                        all_source_mask_b = torch.tensor([f.source_mask for f in eval_features_b], dtype=torch.long)
                        eval_data_b = TensorDataset(all_source_ids_b,all_source_mask_b)
                        dev_dataset['dev_bleu']=eval_examples_b,eval_data_b

                    eval_sampler_b = SequentialSampler(eval_data_b)
                    eval_dataloader_b = DataLoader(eval_data_b, sampler=eval_sampler_b, batch_size=args.eval_batch_size)

                    logger.info("\n***** Running BLEU evaluation on Dev Set *****")
                    logger.info("  Num examples = %d", len(eval_examples_b))

                    model.eval()
                    p_dev=[]
                    for batch_bleu_dev in eval_dataloader_b:
                        batch_bleu_dev = tuple(t.to(device) for t in batch_bleu_dev)
                        source_ids_b,source_mask_b= batch_bleu_dev
                        with torch.no_grad():
                            preds = model(source_ids=source_ids_b,source_mask=source_mask_b)
                            for pred in preds:
                                t=pred[0].cpu().numpy()
                                t=list(t)
                                if tokenizer.eos_token_id in t: t=t[:t.index(tokenizer.eos_token_id)]
                                if t and t[0] == tokenizer.bos_token_id: t = t[1:]
                                elif t and t[0] == tokenizer.cls_token_id: t = t[1:]
                                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                                p_dev.append(text)
                    model.train()
                    predictions_dev=[]
                    dev_output_path = os.path.join(args.output_dir,"dev.output")
                    dev_gold_path = os.path.join(args.output_dir,"dev.gold")
                    with open(dev_output_path,'w', encoding='utf-8') as f_do, open(dev_gold_path,'w', encoding='utf-8') as f_dg:
                        for ref,gold in zip(p_dev,eval_examples_b):
                            predictions_dev.append(str(gold.idx)+'\t'+ref)
                            f_do.write(str(gold.idx)+'\t'+ref+'\n')
                            f_dg.write(str(gold.idx)+'\t'+gold.target+'\n')
                    
                    try:
                        if os.path.exists(dev_gold_path) and predictions_dev:
                            (goldMap_dev, predictionMap_dev) = bleu.computeMaps(predictions_dev, dev_gold_path)
                            dev_bleu_score=round(bleu.bleuFromMaps(goldMap_dev, predictionMap_dev)[0],2)
                            logger.info("  %s = %s "%("dev_bleu-4",str(dev_bleu_score)))
                        else:
                            dev_bleu_score = 0.0
                            logger.warning("Could not compute dev BLEU. Gold file or predictions missing.")
                    except Exception as e:
                        logger.error(f"Error calculating dev BLEU: {e}", exc_info=True)
                        dev_bleu_score = 0.0

                    logger.info("  "+"*"*20)
                    if dev_bleu_score > best_bleu:
                        logger.info("  Best dev bleu improved to %s", dev_bleu_score)
                        best_bleu = dev_bleu_score
                        output_dir_best_bleu = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_best_bleu): os.makedirs(output_dir_best_bleu)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir_best_bleu, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info(f"Saved best BLEU model to {output_model_file}")
                    # --- Kết thúc Evaluation Logic ---

                # --- (1) SAVE CHECKPOINT SAU MỖI args.train_steps (NẾU DEFINED) HOẶC CUỐI EPOCH (NẾU LẶP THEO EPOCH)
                # Điều chỉnh logic save checkpoint này nếu bạn muốn lưu theo step hoặc epoch
                if args.local_rank in [-1, 0] and args.max_steps <=0 : # Chỉ lưu cuối epoch nếu không dùng max_steps
                    # Đã save checkpoint-last và checkpoint-best-ppl/bleu ở trên
                    logger.info(f"End of epoch {epoch_or_step}. Checkpoints saved if metrics improved.")
            # Kết thúc vòng lặp step hoặc epoch
            if args.max_steps > 0 and global_step >= args.max_steps:
                 break # Dừng training nếu đã đạt max_steps

    # --- Phần Test (Thêm Debugging và sửa lỗi tính EM) ---
    if args.do_test:
        print("DEBUG: Entered do_test block.", flush=True)
        files=[]
        if args.dev_filename is not None and os.path.exists(args.dev_filename):
            logger.info(f"Adding dev file for testing: {args.dev_filename}")
            files.append(args.dev_filename)
        if args.test_filename is not None and os.path.exists(args.test_filename):
            logger.info(f"Adding test file for testing: {args.test_filename}")
            files.append(args.test_filename)

        if not files:
            logger.warning("No valid dev or test files provided for do_test. Skipping test phase.")
        else:
            for file_idx, file_path in enumerate(files): # Đổi tên biến
                logger.info("***** Running testing on %s *****", file_path)
                print(f"DEBUG: Testing on file: {file_path}", flush=True)

                eval_examples = read_examples(file_path)
                if not eval_examples:
                    logger.error(f"Could not read examples from {file_path}. Skipping testing for this file.")
                    continue

                eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask)

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = [] # Danh sách các dự đoán text tốt nhất
                print(f"DEBUG: Starting evaluation loop for file: {file_path}", flush=True)
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval bleu for test set {file_idx}"):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask) # Giả định model trả về list of list of tensors
                        if not isinstance(preds, list):
                             logger.warning(f"Model output 'preds' is not a list (type: {type(preds)}). Attempting to handle for batch.")
                             # Cố gắng xử lý, ví dụ nếu nó trả về một tensor duy nhất chứa tất cả beams
                             if isinstance(preds, torch.Tensor) and preds.ndim == 2: # batch_size, num_beams * seq_len? Hoặc 1 beam
                                 # Cần điều chỉnh dựa trên cấu trúc thực tế của Seq2Seq
                                 # Tạm thời bỏ qua batch này nếu cấu trúc không rõ ràng
                                 logger.error("Cannot process non-list preds of this shape. Skipping batch.")
                                 # Thêm placeholders vào p để duy trì độ dài
                                 for _ in range(source_ids.size(0)): p.append("")
                                 continue
                             else: # Các trường hợp khác
                                 logger.error("Cannot process non-list preds. Skipping batch.")
                                 for _ in range(source_ids.size(0)): p.append("")
                                 continue


                        for pred_beams_for_one_item in preds: # pred_beams_for_one_item là list các beam cho một input
                            best_candidate_text = ""
                            if isinstance(pred_beams_for_one_item, list) and pred_beams_for_one_item:
                                first_beam_output_tensor = pred_beams_for_one_item[0] # Lấy beam đầu tiên
                                if isinstance(first_beam_output_tensor, torch.Tensor):
                                    t = first_beam_output_tensor.cpu().numpy()
                                    t = list(t)
                                    # Cắt bỏ padding và token đặc biệt
                                    if tokenizer.eos_token_id in t:
                                        t = t[:t.index(tokenizer.eos_token_id)]
                                    if t and t[0] == tokenizer.bos_token_id : #Thường CLS id cho encoder, BOS cho decoder
                                        t = t[1:]
                                    elif t and t[0] == tokenizer.cls_token_id: #Một số model có thể dùng cls_token như bos
                                        t = t[1:]
                                    best_candidate_text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                                else:
                                    logger.warning(f"Beam output is not a tensor, but {type(first_beam_output_tensor)}")
                            else:
                                logger.warning(f"Prediction for an item is not a list or is empty: {pred_beams_for_one_item}")
                            p.append(best_candidate_text)
                print(f"DEBUG: Finished evaluation loop for {file_path}. Generated predictions: {len(p)}", flush=True)

                if len(p) != len(eval_examples):
                    logger.error(f"CRITICAL: Mismatch in predictions ({len(p)}) and examples ({len(eval_examples)}) for {file_path}. Metrics will be incorrect.")
                    # Không nên continue ở đây nếu vẫn muốn thử tính toán dựa trên những gì có
                else:
                    logger.info(f"Number of predictions matches number of examples for {file_path}.")


                # Ghi file output và gold
                file_basename = os.path.basename(file_path).rsplit('.', 1)[0] # Lấy tên file không có extension
                output_file_path = os.path.join(args.output_dir, f"{file_basename}.output")
                gold_file_path = os.path.join(args.output_dir, f"{file_basename}.gold")

                print(f"DEBUG: Writing output to {output_file_path} and gold to {gold_file_path}...", flush=True)
                predictions_for_bleu_compute = [] # Danh sách cho hàm bleu.computeMaps
                try:
                    with open(output_file_path,'w', encoding='utf-8') as f_out, \
                         open(gold_file_path,'w', encoding='utf-8') as f_gold:
                        for pred_text, gold_example in zip(p, eval_examples):
                            # Ghi best candidate (từ p) vào file output
                            f_out.write(str(gold_example.idx) + '\t' + pred_text.strip() + '\n') # Thêm strip()
                            f_gold.write(str(gold_example.idx) + '\t' + gold_example.target.strip() + '\n') # Thêm strip()
                            predictions_for_bleu_compute.append(str(gold_example.idx) + '\t' + pred_text.strip())
                    print(f"DEBUG: Finished writing files for {file_path}.", flush=True)
                except IOError as e:
                    print(f"ERROR writing output/gold files for {file_path}: {e}", flush=True)
                    logger.error(f"Error writing files for {file_path}: {e}", exc_info=True)
                    continue # Bỏ qua tính toán metrics cho file này nếu ghi lỗi

                # Tính BLEU
                print(f"DEBUG: Computing BLEU for {file_path}...", flush=True)
                current_bleu_score = 0.0
                if os.path.exists(gold_file_path) and predictions_for_bleu_compute:
                    try:
                        (goldMap, predictionMap) = bleu.computeMaps(predictions_for_bleu_compute, gold_file_path)
                        current_bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                        print(f"DEBUG: BLEU computed for {file_path}: {current_bleu_score}", flush=True)
                    except Exception as e:
                        print(f"ERROR during BLEU computation for {file_path}: {e}", flush=True)
                        logger.error(f"ERROR during BLEU computation for {file_path}:", exc_info=True)
                        current_bleu_score = 0.0
                elif not os.path.exists(gold_file_path):
                    logger.error(f"Gold file {gold_file_path} not found for BLEU calculation.")
                else:
                    logger.error(f"Prediction list is empty for BLEU calculation on {file_path}.")


                # Tính EM
                print(f"DEBUG: Computing EM for {file_path}...", flush=True)
                match_count = 0
                if len(p) == len(eval_examples): # Chỉ tính nếu số lượng khớp
                    for pred_text, gold_example in zip(p, eval_examples):
                        pred_clean = pred_text.strip()
                        gold_clean = gold_example.target.strip()
                        if pred_clean == gold_clean:
                            match_count += 1
                    exact_match_score = (match_count / len(eval_examples)) * 100 if len(eval_examples) > 0 else 0.0
                else:
                    logger.warning(f"Skipping EM calculation for {file_path} due to prediction/example count mismatch.")
                    exact_match_score = 0.0 # Hoặc None để biểu thị không tính được
                print(f"DEBUG: EM computed for {file_path}: {exact_match_score:.2f}", flush=True)

                # --- In kết quả cuối cùng ---
                print(f"DEBUG: Logging final results for {file_path}...", flush=True)
                logger.info("***** Results for %s *****", file_path)
                logger.info("  %s = %s", "BLEU-4", str(current_bleu_score))
                logger.info("  %s = %.2f%%", "EM", exact_match_score)
                logger.info("  "+"*"*20)
                print(f"DEBUG: Finished logging final results for {file_path}.", flush=True)


if __name__ == "__main__":
    main()

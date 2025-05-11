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
import bleu
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
from model import Seq2Seq
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

            # Use 'idx' from json if available and it's an int, otherwise use enumerate idx
            try:
                example_idx = int(js.get('idx', idx))
            except ValueError:
                logger.warning(f"Invalid 'idx' value {js.get('idx')} at line {idx+1} in {filename}. Using enumerate index {idx}.")
                example_idx = idx

            code_tokens = js.get('code_tokens', [])
            if not isinstance(code_tokens, list):
                logger.warning(f"Invalid 'code_tokens' format for idx {example_idx} in {filename}. Expected list, got {type(code_tokens)}. Using empty list.")
                code_tokens = []

            docstring_tokens = js.get('docstring_tokens', [])
            if not isinstance(docstring_tokens, list):
                logger.warning(f"Invalid 'docstring_tokens' format for idx {example_idx} in {filename}. Expected list, got {type(docstring_tokens)}. Using empty list.")
                docstring_tokens = []

            code=' '.join(code_tokens).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(docstring_tokens).replace('\n','')
            nl=' '.join(nl.strip().split())
            examples.append(
                Example(
                        idx = example_idx,
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
                 target_ids, # Note: target_ids are for labels (training/PPL), not direct decoder input for model.py's inference
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

        #target (mainly for labels in training, or PPL calculation)
        # model.py's Seq2Seq.forward with target_ids=None handles its own generation start.
        if stage=="test": # For testing, target_ids are not fed to model's generation loop
            target_tokens_for_feature = [] # Dummy, as model.py won't use it for generation
            target_ids_for_feature = [tokenizer.pad_token_id] * args.max_target_length # Placeholder
            target_mask_for_feature = [0] * args.max_target_length
        else: # For train or dev (PPL eval)
            target_tokens_for_feature = tokenizer.tokenize(example.target)[:args.max_target_length-2]
            # model.py's Seq2Seq.forward (training branch) expects target_ids that are shifted labels
            # The processing here is for creating labels for the CrossEntropyLoss
            # The model itself handles the SOS token implicitly or via its Beam class for generation
            # For Roberta, typically CLS is SOS, SEP is EOS.
            # model.py's Seq2Seq takes sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id
            # For training labels, decoder input should start with SOS, and labels should be original target
            
            # Create labels (target_ids for loss)
            # These should align with how model.py calculates loss
            # model.py's training branch:
            # tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            # shift_labels = target_ids[..., 1:].contiguous()
            # This implies target_ids passed to model.py should be like [SOS, token1, token2, ..., EOS, PAD...]
            # And then it shifts for prediction.
            
            # So, target_ids for features should be [SOS, token1, ..., EOS, PAD...]
            processed_target_tokens = [tokenizer.cls_token] + target_tokens_for_feature + [tokenizer.sep_token]
            target_ids_for_feature = tokenizer.convert_tokens_to_ids(processed_target_tokens)
            target_mask_for_feature = [1] * len(target_ids_for_feature)
            padding_length_target = args.max_target_length - len(target_ids_for_feature)
            target_ids_for_feature += [tokenizer.pad_token_id] * padding_length_target
            target_mask_for_feature += [0] * padding_length_target

        if example_index < 1 and stage=='train': # Log only for the first example in train
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))

            logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

            if stage != 'test':
                logger.info("target_tokens (for feature): {}".format([x.replace('\u0120','_') for x in target_tokens_for_feature]))
                logger.info("target_ids (for feature/label): {}".format(' '.join(map(str, target_ids_for_feature))))
                logger.info("target_mask (for feature/label): {}".format(' '.join(map(str, target_mask_for_feature))))

        features.append(
            InputFeatures(
                 example.idx, # Use original example.idx
                 source_ids,
                 target_ids_for_feature,
                 source_mask,
                 target_mask_for_feature,
            )
        )
    return features


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory")
    parser.add_argument("--load_model_path", default=None, type=str, help="Path to trained model" )
    parser.add_argument("--train_filename", default=None, type=str, help="The train filename")
    parser.add_argument("--dev_filename", default=None, type=str, help="The dev filename")
    parser.add_argument("--test_filename", default=None, type=str, help="The test filename")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=256, type=int, help="Max source sequence length") # Increased default
    parser.add_argument("--max_target_length", default=128, type=int, help="Max target sequence length") # Increased default
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.") # Increased default
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: total training steps. Overrides num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int, help="Run evaluation every X steps (global steps).")
    parser.add_argument("--train_steps", default=-1, type=int, help="Alias for max_steps (for compatibility).")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda and torch.cuda.is_available() else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    set_seed(args)
    if args.local_rank in [-1, 0]:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    # Decoder Layer: model.py permutes, so no batch_first=True here
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    # Get num_decoder_layers from config if available, else default (e.g., 6 or 12)
    # model.py currently hardcodes decoder layers in its internal structure (implicitly if not passed)
    # For consistency, let's assume model.py's decoder structure matches encoder or a common value like 6 or 12.
    # If model.py's decoder has a fixed number of layers (e.g. 6), use that. Otherwise, config.num_hidden_layers is a common choice.
    num_decoder_layers = getattr(config, "num_decoder_layers", 6) # Default to 6 if not in config, or match encoder
    logger.info(f"Using {num_decoder_layers} layers for the Transformer Decoder.")
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    model = Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id) # model.py uses sep_token_id as eos

    if args.load_model_path is not None:
        logger.info("Reloading model from {}".format(args.load_model_path))
        device_to_load_on = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() else device
        logger.info(f"Attempting to load model to {device_to_load_on}.")
        try:
            model.load_state_dict(torch.load(args.load_model_path, map_location=device_to_load_on))
            logger.info("Model loaded successfully (direct load).")
        except RuntimeError as e:
            logger.warning(f"Failed to load model directly: {e}. Trying alternatives.")
            try:
                logger.info("Retrying with weights_only=True.")
                model.load_state_dict(torch.load(args.load_model_path, map_location=device_to_load_on, weights_only=True))
                logger.info("Model loaded successfully (weights_only=True).")
            except Exception as e_weights_only:
                logger.warning(f"Failed to load with weights_only=True: {e_weights_only}. Trying to remove 'module.' prefix.")
                try:
                    state_dict = torch.load(args.load_model_path, map_location=device_to_load_on)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                    logger.info("Model loaded successfully (after removing 'module.' prefix).")
                except Exception as e_module:
                    logger.error(f"Failed to load by removing 'module.' prefix: {e_module}. Model loading failed.")
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True) # Set to True if model has unused params
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # =====================================================================================
        # <<<<<<<<<<<<<<<<<<< START: PASTE FULL do_train LOGIC HERE >>>>>>>>>>>>>>>>>>>>>>>
        #  PASTE THE FULL `if args.do_train:` block from the previous complete `run2_12.py`
        #  (the one that included PPL and BLEU evaluation during training).
        #  Make sure to adapt the model output processing in the BLEU eval part of do_train
        #  to match the new way of handling model.py's beam search output, similar to do_test.
        #
        # Example snippet for BLEU eval within do_train (needs full context):
        #
        #   if args.local_rank in [-1, 0] and args.do_eval and ...:
        #       ... (PPL eval) ...
        #       if 'dev_bleu' in dev_dataset: ...
        #       else:
        #           eval_examples_bleu = read_examples(args.dev_filename)
        #           ... (create eval_data_bleu for source_ids, source_mask only) ...
        #       model.eval()
        #       p_bleu=[]
        #       for batch_bleu in tqdm(eval_dataloader_bleu, ... desc="Generating for BLEU (dev)"):
        #           source_ids_b, source_mask_b = tuple(t.to(device) for t in batch_bleu)
        #           with torch.no_grad():
        #               preds_b_beams = model(source_ids=source_ids_b, source_mask=source_mask_b) # No target_ids
        #               for i_beam_dev in range(preds_b_beams.size(0)):
        #                   best_beam_ids_dev = preds_b_beams[i_beam_dev, 0, :].cpu().numpy()
        #                   token_ids_list_dev = []
        #                   # Logic to process best_beam_ids_dev into token_ids_list_dev (same as in do_test)
        #                   for token_id_val_dev in best_beam_ids_dev:
        #                       if token_id_val_dev == tokenizer.eos_token_id:
        #                           token_ids_list_dev.append(token_id_val_dev)
        #                           break
        #                       if token_id_val_dev == 0: # CLS/SOS or model.py padding
        #                           if not token_ids_list_dev and token_id_val_dev == tokenizer.cls_token_id:
        #                               token_ids_list_dev.append(token_id_val_dev)
        #                               continue
        #                           elif token_ids_list_dev: # Padding
        #                               break
        #                           elif not token_ids_list_dev and token_id_val_dev != tokenizer.cls_token_id: # 0 but not CLS at start
        #                               break
        #                       if token_id_val_dev == tokenizer.pad_token_id: # Roberta pad_token_id (usually 1)
        #                           break
        #                       token_ids_list_dev.append(token_id_val_dev)
        #                   text_dev = tokenizer.decode(token_ids_list_dev, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #                   p_bleu.append(text_dev)
        #       model.train()
        #       # ... (rest of BLEU calculation from p_bleu and eval_examples_bleu) ...
        # =====================================================================================
        logger.warning("Full 'do_train' logic needs to be pasted here from a previous version and adapted.")
        pass # REMOVE THIS PASS AND PASTE THE do_train LOGIC


    if args.do_test:
        files_to_test=[]
        if args.dev_filename is not None and os.path.exists(args.dev_filename):
            files_to_test.append(args.dev_filename)
            logger.info(f"Added dev file for testing: {args.dev_filename}")
        elif args.dev_filename:
            logger.warning(f"Dev file {args.dev_filename} not found, will not be tested.")

        if args.test_filename is not None and os.path.exists(args.test_filename):
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
            eval_data = TensorDataset(all_source_ids,all_source_mask) # Only source for inference

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

            model.eval()
            p_test = []
            logger.info(f"Starting prediction generation for {file_basename}...")
            for batch_data in tqdm(eval_dataloader,total=len(eval_dataloader), desc=f"Predicting for {file_basename}"):
                # Batch data only contains source_ids and source_mask for testing
                source_ids_test, source_mask_test = tuple(t.to(device) for t in batch_data)
                with torch.no_grad():
                    batch_preds_beams = model(source_ids=source_ids_test, source_mask=source_mask_test)
                    # batch_preds_beams shape: [batch_size, beam_size, max_length]

                    for i in range(batch_preds_beams.size(0)):
                        best_beam_ids = batch_preds_beams[i, 0, :].cpu().numpy() # Top beam

                        token_ids_for_decode = []
                        for token_id_val in best_beam_ids:
                            # model.py's Beam.buildTargetTokens doesn't include SOS.
                            # model.py's Seq2Seq.forward pads with zero tensor.
                            # Roberta: cls_token_id=0 (SOS for model.py), sep_token_id=2 (EOS for model.py), pad_token_id=1
                            
                            if token_id_val == tokenizer.sep_token_id: # EOS used by model.py's Beam
                                token_ids_for_decode.append(token_id_val) # Include EOS for skip_special_tokens
                                break
                            
                            # Check for padding 0 from model.py's zero tensor
                            # This should only happen AFTER some actual tokens, not if 0 is the SOS/CLS token
                            if token_id_val == 0 and token_ids_for_decode: # If 0 appears after other tokens, it's padding
                                break
                            # If 0 is the very first token and it's not the cls_token (unlikely scenario)
                            elif token_id_val == 0 and not token_ids_for_decode and token_id_val != tokenizer.cls_token_id:
                                break
                            
                            # Roberta's actual pad_token_id is 1. If model.py somehow used it, stop.
                            if token_id_val == tokenizer.pad_token_id:
                                break
                                
                            token_ids_for_decode.append(token_id_val)
                        
                        # skip_special_tokens=True should handle removing CLS (0), SEP (2), PAD (1)
                        text = tokenizer.decode(token_ids_for_decode, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        p_test.append(text)
            logger.info(f"Finished prediction generation for {file_basename}. Generated {len(p_test)} predictions.")

            if len(p_test) != len(eval_examples):
                logger.error(f"Mismatch in number of predictions ({len(p_test)}) and examples ({len(eval_examples)}) for file {current_file_to_test}. Skipping EM and BLEU.")
                continue

            exact_matches_test = []
            predictions_for_bleu_test_file = []
            output_file_path_test = os.path.join(args.output_dir, f"test_{file_idx}_{os.path.splitext(file_basename)[0]}.output") # Use file_idx and cleaned basename
            gold_file_path_test = os.path.join(args.output_dir, f"test_{file_idx}_{os.path.splitext(file_basename)[0]}.gold")

            logger.info(f"Writing predictions to: {output_file_path_test}")
            logger.info(f"Writing gold references to: {gold_file_path_test}")

            with open(output_file_path_test, 'w', encoding='utf-8') as f_out, \
                 open(gold_file_path_test, 'w', encoding='utf-8') as f_gold:
                for i_pred, (pred_text, gold_example) in enumerate(zip(p_test, eval_examples)):
                    normalized_pred_em = pred_text.strip().lower()
                    normalized_gold_em = gold_example.target.strip().lower()
                    is_em = (normalized_pred_em == normalized_gold_em)
                    exact_matches_test.append(is_em)

                    f_out.write(str(gold_example.idx) + '\t' + pred_text.replace('\n', ' ').strip() + '\n')
                    f_gold.write(str(gold_example.idx) + '\t' + gold_example.target.replace('\n', ' ').strip() + '\n')
                    predictions_for_bleu_test_file.append(str(gold_example.idx) + '\t' + pred_text.replace('\n', ' ').strip())
            logger.info(f"Finished writing files for {file_basename}.")

            if exact_matches_test:
                em_score_test = np.mean(exact_matches_test) * 100
                num_em_correct_test = sum(exact_matches_test)
                logger.info(f"  Exact Match (EM) for {file_basename} = {em_score_test:.2f}% ({num_em_correct_test}/{len(exact_matches_test)})")
            else:
                logger.info(f"  Exact Match (EM) for {file_basename} = N/A (no predictions to compare)")
            logger.info("  " + "*" * 20)

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
        logger.info("***** Testing finished *****")

if __name__ == "__main__":
    main()

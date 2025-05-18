# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# ... (các dòng license và import giữ nguyên) ...
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
                        idx = js['idx'], 
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
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        if stage=="test":
            # For test, target_ids are not strictly needed for model input if model.py handles generation start
            # But for consistency or if some part of your pipeline expects it, create dummy ones.
            # model.py's Seq2Seq.forward handles generation internally when target_ids is None
            target_tokens = [] # Or tokenizer.tokenize("None") if needed for some reason
            target_ids = [tokenizer.pad_token_id] * args.max_target_length # Dummy, won't be used by Seq2Seq.forward if target_ids is None
            target_mask = [0] * args.max_target_length
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
            # For training/evaluation (PPL), target_ids are labels
            # model.py's Seq2Seq.forward expects target_ids for training
            processed_target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token] # model.py không thêm cls, sep cho target
            target_ids = tokenizer.convert_tokens_to_ids(processed_target_tokens)
            target_mask = [1] *len(target_ids)
            padding_length_target = args.max_target_length - len(target_ids)
            target_ids+=[tokenizer.pad_token_id]*padding_length_target
            target_mask+=[0]*padding_length_target


        if example_index < 5 and stage=='train':
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
            if stage != 'test':
                logger.info("target_tokens (original for loss): {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids (for loss/ppl): {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask (for loss/ppl): {}".format(' '.join(map(str, target_mask))))

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    # ... ( giữ nguyên tất cả các parser.add_argument ) ...
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )
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
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int, help="Number of update steps between two evaluations.")
    parser.add_argument("--train_steps", default=-1, type=int, help="Total number of training steps to perform (overrides num_train_epochs).")
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
    # Sửa đổi: Bỏ batch_first=True vì model.py tự permute
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    num_decoder_layers = getattr(config, "num_decoder_layers", 12) # Lấy từ config nếu có, mặc định 12
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    model = Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

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
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # ... (Giữ nguyên phần do_train như trong file đã sửa ở các phản hồi trước)
        # Đảm bảo phần eval_bleu trong do_train cũng được cập nhật cách xử lý output model:
        # Ví dụ:
        # if args.local_rank in [-1, 0] and args.do_eval and args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_flag:
        #    ... (PPL eval) ...
        #    ... (BLEU eval for dev set)
        #        model.eval()
        #        p_bleu=[]
        #        for batch_bleu in tqdm(eval_dataloader_bleu, ...):
        #            ...
        #            preds_b_beams = model(source_ids=source_ids_b, source_mask=source_mask_b) # Không có target_ids
        #            for i_beam_dev in range(preds_b_beams.size(0)):
        #                best_beam_ids_dev = preds_b_beams[i_beam_dev, 0, :].cpu().numpy()
        #                # ... (decode logic như trong do_test) ...
        #                p_bleu.append(text_dev)
        #        model.train()
        #        # ... (tính BLEU từ p_bleu) ...
        pass # Placeholder, bạn cần điền lại logic training hoàn chỉnh


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

            # Quan trọng: khi do_test, target_ids không được sử dụng để truyền vào model.forward
            # mà model.py sẽ tự chạy beam search.
            # convert_examples_to_features sẽ tạo target_ids dummy nếu stage='test'
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            # Không cần all_target_ids, all_target_mask cho input vào DataLoader ở đây vì model.py không dùng
            eval_data = TensorDataset(all_source_ids,all_source_mask)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

            model.eval()
            p_test = []
            logger.info(f"Starting prediction generation for {file_basename}...")
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader), desc=f"Predicting for {file_basename}"):
                batch_tuple = tuple(t.to(device) for t in batch)
                source_ids_test, source_mask_test = batch_tuple # Chỉ có source_ids và source_mask
                with torch.no_grad():
                    # model.py's forward handles beam search when target_ids is None
                    batch_preds_beams = model(source_ids=source_ids_test, source_mask=source_mask_test)
                    # batch_preds_beams shape: [batch_size, beam_size, max_length]

                    for i in range(batch_preds_beams.size(0)): # Iterate through samples in the batch
                        best_beam_ids = batch_preds_beams[i, 0, :].cpu().numpy() # Get the top beam (index 0)
                        
                        token_ids_list = []
                        for token_id in best_beam_ids:
                            if token_id == tokenizer.eos_token_id:
                                break
                            # model.py's Beam class pads with zero if prediction is shorter than max_length
                            # tokenizer.pad_token_id is often 0 for Roberta, but check just in case
                            if token_id == tokenizer.pad_token_id and token_id != tokenizer.sos_token_id : # Avoid break if pad is sos (unlikely)
                                break
                            if token_id == tokenizer.sos_token_id: # Usually, SOS is not part of the final output
                                continue
                            token_ids_list.append(token_id)
                        
                        text = tokenizer.decode(token_ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        p_test.append(text)
            logger.info(f"Finished prediction generation for {file_basename}. Generated {len(p_test)} predictions.")

            if len(p_test) != len(eval_examples):
                logger.error(f"Mismatch in number of predictions ({len(p_test)}) and examples ({len(eval_examples)}) for file {current_file_to_test}. Skipping EM and BLEU.")
                continue

            exact_matches_test = []
            predictions_for_bleu_test_file = []
            output_file_path_test = os.path.join(args.output_dir, f"test_{file_idx}_{file_basename}.output")
            gold_file_path_test = os.path.join(args.output_dir, f"test_{file_idx}_{file_basename}.gold")

            logger.info(f"Writing predictions to: {output_file_path_test}")
            logger.info(f"Writing gold references to: {gold_file_path_test}")

            with open(output_file_path_test, 'w', encoding='utf-8') as f_out, \
                 open(gold_file_path_test, 'w', encoding='utf-8') as f_gold:
                for i_pred, (pred_text, gold_example) in enumerate(zip(p_test, eval_examples)):
                    normalized_pred_em = pred_text.strip().lower()
                    normalized_gold_em = gold_example.target.strip().lower()
                    is_em = (normalized_pred_em == normalized_gold_em)
                    exact_matches_test.append(is_em)

                    # For BLEU files, usually keep original casing unless specified
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

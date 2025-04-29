mkdir -p model/code2review_t5_data_task2

CUDA_VISIBLE_DEVICES=0 python run_12.py --do_train --do_eval --model_type roberta      \
     	--model_name_or_path microsoft/codebert-base   \
 	    --train_filename "/kaggle/input/daataa5/task2_data/t5_data/train.json"       \
     	--dev_filename "/kaggle/input/daataa5/task2_data/t5_data/val.json"      \
     	--output_dir model/code2review_t5_data_task2/   \
 	    --max_source_length 512 --max_target_length 100  --beam_size 5   --train_batch_size 16  --eval_batch_size 16        \
      --learning_rate 5e-5   --train_steps 50000  --eval_steps 5000

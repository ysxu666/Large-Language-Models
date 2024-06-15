微调指令

python src/sft_finetune.py --model_name_or_path path/to/pretrained/model --train_file path/to/train.jsonl --output_dir path/to/output_dir --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 10000 --logging_dir ./logs

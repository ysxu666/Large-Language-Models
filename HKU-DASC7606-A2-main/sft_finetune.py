import os
import argparse
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from configuration_phi import PhiConfig
from modeling_phi import PhiForCausalLM
from tokenization_codegen import CodeGenTokenizer

# 定义脚本的参数
parser = argparse.ArgumentParser(description='微调Phi模型')
parser.add_argument('--model_name_or_path', type=str, required=True, help='预训练模型的路径')
parser.add_argument('--train_file', type=str, required=True, help='训练数据文件路径（jsonl格式）')
parser.add_argument('--output_dir', type=str, required=True, help='微调后模型的输出目录')
parser.add_argument('--num_train_epochs', type=int, default=3, help='训练的轮数')
parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='每个设备上的训练批次大小')
parser.add_argument('--save_steps', type=int, default=10_000, help='每X步保存一次检查点')
parser.add_argument('--logging_dir', type=str, default='./logs', help='存储日志的目录')

args = parser.parse_args()

# 加载预训练模型和分词器
config = PhiConfig.from_pretrained(args.model_name_or_path)
model = PhiForCausalLM.from_pretrained(args.model_name_or_path, config=config)
tokenizer = CodeGenTokenizer.from_pretrained(args.model_name_or_path)

# 定义数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# 加载和预处理数据集
dataset = load_dataset('json', data_files=args.train_file, split='train')
dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# 为语言模型训练创建数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    save_steps=args.save_steps,
    save_total_limit=2,
    logging_dir=args.logging_dir,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model(args.output_dir)

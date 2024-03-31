import argparse
import pprint
import os
import copy
from str2bool import str2bool
from typing import Dict, Sequence
from sentence_transformers import SentenceTransformer

IGNORE_INDEX = -100

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--device_id', type=str, default="")
parser.add_argument('--model', type=str, default='', help="")
parser.add_argument('--embedder', type=str, default="")
parser.add_argument('--output_path', type=str, help="")
parser.add_argument('--start_index', type=int, default=0, help="")
parser.add_argument('--end_index', type=int, default=164, help="")
parser.add_argument('--N', type=int, default=8, help="")
parser.add_argument('--max_len', type=int, default=512, help="")
parser.add_argument('--overwrite', type=str2bool, default=False, help="")
parser.add_argument('--prompt_type', type=str, default="v1.0", help="")
parser.add_argument('--top_k', type=str2bool, default=False, help="")
parser.add_argument('--top_k_reverse', type=str2bool, default=False, help="")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)


from tqdm import tqdm
import torch
import json

import transformers
from modeling_phi import PhiForCausalLM
from tokenization_codegen import CodeGenTokenizer


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_arc_problems(data_path="data/ARC-Easy-test.jsonl"):
    dataset = []
    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            candidate_answers = " ".join([f"({label}) {text}" for text, label in zip(json_obj["choices"]["text"], json_obj["choices"]["label"])]).strip()
            for text, label in zip(json_obj["choices"]["text"], json_obj["choices"]["label"]):
                dataset.append({
                    "id": json_obj["id"],
                    "question": json_obj["question"],
                    "candidate_answers": candidate_answers,
                    "answer": text,
                    "label": label,
                    "answerKey": json_obj["answerKey"],
                })
    print(f"Dataset size: {len(dataset)}")

    return dataset


def load_all_demonstrations(train_path="data/ARC-Challenge-train.jsonl"):
    demonstrations = []
    with open(train_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            demonstrations.append((json_obj["question"], json_obj["choices"]["text"], json_obj["choices"]["label"], json_obj["answerKey"]))
    print(f"load {len(demonstrations)} demonstrations from {train_path}")
    return demonstrations


def llm_embedder(llm, sentences, is_query=True):
    INSTRUCTIONS = {
        "qa": {
            "query": "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "icl": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "chat": {
            "query": "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query": "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "tool": {
            "query": "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
        "convsearch": {
            "query": "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
    }

    instruction = INSTRUCTIONS["icl"]
    if is_query:
        sentences = [instruction["query"] + s for s in sentences]
    else:
        sentences = [instruction["key"] + s for s in sentences]

    # Encode
    sentence_embeddings = llm.encode(sentences)
    print(f"Sentence embeddings size: {sentence_embeddings.shape}")
    return sentence_embeddings

def candidate_answers_formating(texts, labels):
    candidate_answers = " ".join([f"({label}) {text}" for text, label in zip(texts, labels)]).strip()
    return candidate_answers

# task 4
def example_formating(question, answer=None, candidate_answers=None, prompt_type="v2.0"):
    exam_instruction = "Instruction: Read the following question carefully and select the most appropriate answer."
    if prompt_type == "v1.0":
        if answer is not None:
            prompt = f"Question: {question}\nCandidate answers: {candidate_answers}\nGold answer: {answer}"
        else:
            prompt = f"Question: {question}\nCandidate answers: {candidate_answers}\nGold answer:"
    elif prompt_type == "v2.0":
        if answer is not None:
            prompt = f"Question: {question}\nOptions: {candidate_answers}\nCorrect Answer: {answer}"
        else:
            prompt = f"Question: {question}\nOptions: {candidate_answers}\nCorrect Answer:"
    elif prompt_type == "v3.0":
        if answer is not None:
            prompt =f"{exam_instruction}\nQuestion: {question}\nOptions: {candidate_answers}\nCorrect Answer: {answer}"
        else:
            prompt = f"{exam_instruction}\nQuestion: {question}\nOptions: {candidate_answers}\nCorrect Answer:"    

    else:
        raise NotImplementedError
    return prompt

# Prompt的生成过程
# 在这个脚本中，"prompt"的生成是一个关键步骤，它直接影响模型的输入和因此的表现。生成"prompt"的主要函数是generate_prompt，它的作用是根据给定的问题和一系列示例（demonstrations），构建一个用于模型输入的文本。

# 1. 函数参数
# question: 当前需要回答的问题。
# candidate_answers: 当前问题的候选答案列表。
# prompt_type: "prompt"的格式类型，影响最终"prompt"的结构。
# N: 在"prompt"中要使用的示例（demonstrations）的数量。
# demonstrations: 一系列示例，每个示例包括问题、候选答案、正确答案等信息。
# demonstration_embeddings: 示例的嵌入表示，用于寻找与当前问题最相似的示例。
# embedder: 用于生成嵌入表示的模型。
# top_k: 是否只选取最相似的N个示例。
# top_k_reverse: 是否反转最相似示例的顺序。
# 2. "Prompt"的构建过程
# 相似性匹配: 如果启用top_k，则首先计算当前问题与所有示例之间的相似度，然后根据相似度选择最相似的N个示例。如果启用top_k_reverse，则会反转所选示例的顺序。

# 格式化: 对于每个选定的示例，使用example_formating函数将问题、候选答案和正确答案格式化成一定格式的文本。这个格式由prompt_type参数控制。

# 组合: 将所有选定示例的文本和当前问题的文本组合起来，形成最终的"prompt"。

# 使用"Prompt"
# 生成的"prompt"将作为模型的输入，模型会基于这个输入生成或预测一个答案。在这个脚本中，generate_prompt函数的输出被用作preprocess函数的输入之一，preprocess函数进一步处理这些文本以适配模型的输入格式。

def generate_prompt(question, candidate_answers, prompt_type, N,
                    demonstrations, demonstration_embeddings, embedder,
                    top_k=False, top_k_reverse=False):

    indices = list(range(len(demonstrations)))
    if top_k: # task 5
        question_embeddings = llm_embedder(embedder, [question], True) # [1, n_dim]
        similarity = question_embeddings @ demonstration_embeddings.T  # [1, n_demo]
        indices_sorted = sorted(list(range(len(demonstrations))), key=lambda x: similarity[0][x], reverse=True)
        if top_k_reverse:
            indices = indices_sorted[:N][::-1] + indices_sorted[N:]
        else:
            indices = indices_sorted

    template = ""
    for idx in indices[:N]:
        demo = demonstrations[idx]
        candidate = candidate_answers_formating(demo[1], demo[2])
        gold = demo[1][demo[2].index(demo[3])]
        template += f"\n\n{example_formating(demo[0], answer=gold, candidate_answers=candidate, prompt_type=prompt_type)}"

    template += f"\n\n{example_formating(question, candidate_answers=candidate_answers, prompt_type=prompt_type)}"

    return template.strip()


def get_model(
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = CodeGenTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = PhiForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=args.max_len,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=torch.stack(input_ids).to(device), labels=torch.stack(labels).to(device))



def main():

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = get_arc_problems(args.data_path)[args.start_index: args.end_index]

    num_samples = len(problems)
    tokenizer, model = get_model(base_model=args.model)
    print(f"Loaded {args.model}.")

    embedder = SentenceTransformer(args.embedder, device=device)
    print(f"loaded {args.embedder}.")

    demonstrations = load_all_demonstrations(args.data_path.replace("test", "train").replace("validation", "train"))
    demonstration_embeddings = llm_embedder(embedder, [d[0] for d in demonstrations], False) # ndarray: [n_demons, n_dim]
    print(f"Demonstration embeddings size: {demonstration_embeddings.shape}")

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        question = problems[i]["question"]
        answer = problems[i]["answer"]
        candidate_answers = problems[i]["candidate_answers"]

        source = generate_prompt(question, candidate_answers, args.prompt_type, args.N,
                                 demonstrations, demonstration_embeddings, embedder,
                                 top_k=args.top_k, top_k_reverse=args.top_k_reverse)
        if i == 0:
            print(f"prompt #{i}: {source}")

        target = " {}".format(answer)
        encoding = preprocess([source], [target], tokenizer)

        with torch.no_grad():
            # task 6
            outputs = model(**encoding)
            log_likelihood = outputs.loss * -1


        print("Saving results to {}".format(output_file))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": problems[i]["id"],
                "log_likelihood": log_likelihood.tolist(),
                "question": question,
                "candidate_answers": candidate_answers,
                "answer": answer,
                "label": problems[i]["label"],
                "answerKey": problems[i]["answerKey"],
            }) + "\n")




if __name__ == '__main__':
    main()

import os
import argparse
import json
# /**
# 这段代码是一个Python脚本，用于评估多选题回答模型的准确度。具体来说，它读取模型生成的预测结果，并计算模型在答案预测上的准确率。下面是代码的详细解读：

# 1. **解析命令行参数**：使用`argparse`库来解析命令行输入的参数。这里它解析一个参数`--prediction_path`，即模型预测结果的路径。

# 2. **读取预测结果**：代码遍历指定路径下的所有文件，每个文件被认为是一个包含预测结果的JSONL（JSON Line）格式文件。对于文件中的每一行：
#    - 使用`json.loads`解析JSON格式的行。
#    - 从每个JSON对象中提取`id`（问题的标识符）、`label`（模型预测的答案）、`answerKey`（正确答案的标识）、和`log_likelihood`（预测答案的对数似然值）。

# 3. **储存预测信息**：预测的标签和对数似然值根据问题的`id`存储在`id_to_score`字典中。正确答案存储在`id_to_answer`字典中。

# 4. **计算准确率**：遍历每个问题的`id`，对每个问题的所有预测答案按照对数似然值从高到低排序，取最高的一个预测答案。如果这个答案与正确答案相匹配，则认为这个问题回答正确。通过比较所有问题的总数和回答正确的问题数来计算准确率。

# 5. **输出结果**：最后，打印出模型在整个预测路径上的总准确率，包括正确回答的数量和总问题数量。

# 这个脚本是在机器学习和自然语言处理任务中常用的评估工具，用于评价模型在特定任务上的性能。
# */
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARC evaluation')
    parser.add_argument('--prediction_path', type=str)

    args = parser.parse_args()
    prediction_path = args.prediction_path
    id_to_score = {}
    id_to_answer = {}
    for file in os.listdir(prediction_path):
        with open(os.path.join(prediction_path, file), encoding="utf-8") as f:
            for line in f.readlines():
                json_obj = json.loads(line)
                id = json_obj["id"]
                label = json_obj["label"]
                answerKey = json_obj["answerKey"]
                log_likelihood = json_obj["log_likelihood"]
                if id not in id_to_score:
                    id_to_score[id] = [(label, log_likelihood)]
                else:
                    id_to_score[id].append((label, log_likelihood))
                if id not in id_to_answer:
                    id_to_answer[id] = answerKey
    correct, total = 0, 0
    for id in id_to_score:
        highest = sorted(id_to_score[id], key=lambda x: x[1], reverse=True)[0]
        if highest[0] == id_to_answer[id]:
            correct += 1
        total += 1
    print("({}) Acc: {} ({} / {})".format(prediction_path, correct / total, correct, total))

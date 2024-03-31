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

#这个JSONL文件中的每个对象包含了对单个问题的评估结果。每个键（key）代表的意义如下：

# 1. **`id`**: 唯一标识符，用于标识特定的问题。在这个例子中，`"Mercury_SC_407689"`是问题的唯一ID。

# 2. **`log_likelihood`**: 对数似然值，表示模型给出正确答案的概率的对数。在机器学习和统计中，对数似然是一个重要的度量，用来评估模型对数据的拟合程度。这个值越高，表示模型认为给定的答案越有可能是正确的。在这个例子中，`-4.67886209487915`是模型给出的对数似然值。

# 3. **`question`**: 问题文本。这是模型需要回答的问题。例如，`"Marshall researched some of the changes on Earth that happen over a long time. Which event most likely causes the slowest change to the surface of Earth?"` 是一个关于地球表面变化的问题。

# 4. **`candidate_answers`**: 候选答案列表，以一定的格式（通常是选项标签和答案文本的组合）给出。在这个例子中，有四个候选答案：`"(A) earthquake (B) landslide (C) volcano (D) wind"`。

# 5. **`answer`**: 实际选择的答案文本。这是模型根据问题内容和候选答案列表给出的答案。注意，这个答案可能不一定是正确的答案。在这个例子中，模型选择了`"earthquake"`作为答案。

# 6. **`label`**: 实际选择的答案的标签。这是模型选择的答案对应的选项标签。在这个例子中，模型选择了标签`"A"`对应的答案。

# 7. **`answerKey`**: 正确答案的标签。这是问题的正确答案对应的选项标签，用于评估模型性能。在这个例子中，正确答案的标签是`"D"`。

# 这些键值提供了对每个问题评估结果的详细描述，包括问题本身、候选答案、模型选择的答案及其对数似然值，以及正确答案的标签，使得可以详细分析模型的性能。

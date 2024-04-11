import os
import argparse
import json
from time import sleep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARC evaluation')
    parser.add_argument('-p', '--prediction_path', type=str)

    args = parser.parse_args()
    prediction_path = args.prediction_path
    id_to_score = {}
    id_to_answer = {}
    for file in os.listdir(prediction_path):
        if file.split('.')[-1] == 'jsonl':
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
    # for keys in id_to_score:
        # print(keys, len(id_to_score[keys]))
    # print(id_to_score)
    for id in id_to_score:
        highest = sorted(id_to_score[id], key=lambda x: x[1], reverse=True)[0]
        print(id, highest[0], id_to_answer[id])
        if highest[0] == id_to_answer[id]:
            correct += 1
        total += 1
    print("({}) Acc: {} ({} / {})".format(prediction_path, correct / total, correct, total))

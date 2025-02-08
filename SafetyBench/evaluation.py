import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
# Sample predictions and true labels
true_labels = []
predictions = []
with open('/home/yyyzwyy/evaluation/data/test_zh_subset_eva_baichuan-chat-13b_zeroshotFalse_res_processed.json', 'r') as f:
    data = json.load(f)
    for value in data.values():
        true_labels.append("("+chr(65 + value)+")")  # 0->A, 1->B, 2->C, 3->D

with open('/home/yyyzwyy/evaluation/data/test_zh_subset_eva_baichuan-chat-13b_zeroshotFalse_res.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        predictions.append(item['origin_pred'])
#Calculate accuracy and F1 score
accuracy = accuracy_score(true_labels,predictions)
f1 =f1_score(true_labels, predictions, average='macro')
print(f'Accuracy:{accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')
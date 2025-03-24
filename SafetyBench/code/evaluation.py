import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
# Sample predictions and true labels
true_labels = []
predictions = []
with open('/home/yyyzwyy/evaluation/data/1.json', 'r') as f:
    pred_data = json.load(f)
    for value in pred_data.values():
        predictions.append(value)

with open('/home/yyyzwyy/evaluation/data/dev_zh.json', 'r') as f:
    dev_data = json.load(f)
    for category in dev_data.values():
        for item in category:
            true_labels.append(item['answer'])
#Calculate accuracy and F1 score
print(true_labels,predictions)
accuracy = accuracy_score(true_labels,predictions)
f1 =f1_score(true_labels, predictions, average='macro')
print(f'Accuracy:{accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')
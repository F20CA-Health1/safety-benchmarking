import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple

def calculate_metrics(true_labels_path: str, predictions_path: str) -> Tuple[float, float]:
    """计算评估指标
    
    Args:
        true_labels_path: 真实标签文件路径
        predictions_path: 预测结果文件路径
        
    Returns:
        Tuple[float, float]: (accuracy, f1_score)
    """
    # 读取真实标签
    true_labels = []
    with open(true_labels_path, 'r') as f:
        data = json.load(f)
        for value in data.values():
            true_labels.append("(" + chr(65 + value) + ")")  # 0->A, 1->B, 2->C, 3->D

    # 读取预测结果
    predictions = []
    with open(predictions_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            predictions.append(item['origin_pred'])

    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return accuracy, f1 
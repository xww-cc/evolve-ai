from typing import List
from config.global_constants import STAGNATION_WINDOW, STAGNATION_THRESHOLD
import numpy as np

def detect_stagnation(history_avg_scores: List[float]) -> bool:
    """停滞检测 - 完整"""
    if len(history_avg_scores) < STAGNATION_WINDOW:
        return False
    recent_avg_scores = history_avg_scores[-STAGNATION_WINDOW:]
    score_change = abs(recent_avg_scores[-1] - np.mean(recent_avg_scores[:-1]))
    return score_change < STAGNATION_THRESHOLD
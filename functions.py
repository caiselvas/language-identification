import math
from typing import Callable, Optional
from nltk.collocations import TrigramCollocationFinder

# SMOOTHING FUNCTIONS

# Functions for Lidstone smoothing
def lidstone_smooth(lambda_value: float, total_trigrams: int, trigram_counts: dict, b_value: int, trigram: tuple):
    number = trigram_counts.get(trigram, 0)
    probs = (number + lambda_value) / (total_trigrams + lambda_value * b_value)
    return probs

def pau_discounting(delta: float, total_trigrams: int, trigram_counts: dict, trigram: tuple, b_value: int = 0):
    number = trigram_counts.get(trigram, 0)
    unique = len(trigram_counts)
    prob = max(number - delta, 0) / total_trigrams + (delta * unique / total_trigrams) * (1 / unique)
    return prob

def absolute_discounting(alpha: float, total_trigrams: int, trigram_counts: dict, b_value: int, trigram: tuple):
    count_trigram = trigram_counts.get(trigram, 0)
    unique = len(trigram_counts)
    if count_trigram == 0:
        prob = ((b_value - unique)*alpha/unique)/total_trigrams
    else:
        prob = ((count_trigram - alpha)/ total_trigrams)
    return prob

def linear_discounting(alpha: float, total_trigrams: int, trigram_counts: dict, b_value: int, trigram: tuple):
    count_trigram = trigram_counts.get(trigram, 0)
    if count_trigram == 0:
        prob = alpha / (b_value - len(trigram_counts))
    else:
        prob = (1-alpha)*(count_trigram / total_trigrams)
    return prob

def probs_total(b_value: int, text: str, model: dict, total_trigrams: int, smooth: Callable = lidstone_smooth, param: float = 0.5, probabilities: Optional[dict] = None):
    trigram_finder = TrigramCollocationFinder.from_words(text)
    prob_sec = 0
    if probabilities == None:
        for trigram, num_instances in trigram_finder.ngram_fd.items():
            prob_sec += num_instances * math.log(smooth(param, trigram=trigram, b_value=b_value, total_trigrams=total_trigrams, trigram_counts=model))
    else:
        for trigram, num_instances in trigram_finder.ngram_fd.items():
            prob_sec += num_instances * probabilities.get(trigram,math.log(smooth(param, trigram=trigram, b_value=b_value, total_trigrams=total_trigrams, trigram_counts=model)))
    return prob_sec

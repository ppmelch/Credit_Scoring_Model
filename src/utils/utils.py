import re
import logging

import pandas as pd
import numpy as np
import warnings 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def print_results(acc, scores_test, model):

    print("Accuracy:", acc)
    print("Scores:", scores_test[:10])
    print(f"Thresholds used → t1: {model.t1}, t2: {model.t2}")
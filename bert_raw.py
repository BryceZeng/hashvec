import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

train = pd.read_excel('1103-1109.xlsx')


sentences = re.sub("[.,!?\\-]", '', train['detail_list'].lower()
                   ).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}


for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)

token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)

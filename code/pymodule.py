import pickle
import sys
import time
import argparse
import math
import os
import shutil
import random
import csv
import torch
import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.datasets import Amazon
from torch_geometric.utils import to_networkx
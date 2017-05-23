"""
Code untuk melakukan learning data
"""
# Import dari code util.py
from util import load_dataset, learning

# Load dataset learning yang sudah melalui feature selection
# Param :
# url = alamat file berada, cth '../Data/train.csv' atau 'train.csv'
# model = alamat tempat model akan disimpan, cth 'logreg_LRmodel.sav' atau 'logreg_LRmodel' 
url = '../Data/feat_sel_K_12.csv'
model = 'logreg_Test_2LRmodel.sav'
dataArray = load_dataset(url) # load dataset
learning(dataArray,model) # learn dataset
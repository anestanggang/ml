"""
Code untuk melakukan feature selection data
"""
# Import dari code util.py
from util import load_dataset,edit_data,train_choosing_feature,selected_feature

# File input berasal dari hasil data crawl
url_raw = "../Data/Test/data_crawl_clean_april.csv"
rawData = load_dataset(url_raw)
# Data raw akan dilakukan penghapusan fitur yang tidak diperlukan dalam seleksi
dataArray = edit_data(rawData)
file_header = 'test_april_header_feature.txt' # Alamat dan nama file list nama header 
file_data = 'test_april_DF_12.csv' # Alamat dan nama file data hasil seleksi
train_choosing_feature(dataArray,file_header,file_data)
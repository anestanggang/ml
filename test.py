"""
Code untuk melakukan learning data
"""
# Import dari code util.py
from util import load_dataset,test_choosing_feature,testing

# Membaca data test raw hasil crawl
url_raw = '../Data/Test/data_crawl_clean_april.csv'
dataArray = load_dataset(url_raw)
file_feature_header = "test_header_feature.txt"
file_new_data = "test_april.csv"
# Penggubahan dataset sesuai dengan hasil seleksi fitur model
new_dataset = test_choosing_feature(dataArray,file_feature_header,file_new_data) 
model = 'logreg_Test_2LRmodel.sav' # Alamat dan nama model hasil learning
file_result = 'Pred.csv' # Alamat dan nama file untuk menyimpan hasil prediksi
testing(new_dataset,model,file_result)
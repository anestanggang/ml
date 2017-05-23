"""
Code yang bekerja sebagai pwnyimpan method yang digunakan dalam
machine learning yang dilakukan.
Method yang tersedia mencakupi dari proses preprocessing fitur,
learning, evaluation, testing.
"""

import pandas as pd
import pickle
import numpy as np
import unicodecsv
import csv
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import chi2, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Fungsi untuk load dataset 
def load_dataset(filename):
    """
    Param :
    filename = string
        Path dan nama file dataset
    """
    dataFrame = pd.read_csv(filename, header = 0)
    return dataFrame

# Fungsi untuk mengubah raw data hasil crawl menjadi fitur-fitur yang diperlukan untuk 
# melakukan seleksi fitur
def edit_data(dataset):
    """
    Param :
    filename = string
        Path dan nama file dataset
    """    
    fitur_extract = ['domain','slogan','name_shop','shop_owner','merchant_status', 'month_op','year_op','shop_age(Month)'
                    ,'UID','sold_product','total_etalase','Active','Responship','Timestamp','slogan_2','location']
    for i in range(len(fitur_extract)):
        dataset = dataset.drop(fitur_extract[i],axis=1)
    return dataset

# Fungsi untuk melihat fitur terbaik dari data train
# Fungsi menggunakan method selectKbest dengan k = 12
# Penggantian K dapat dilakukan sesuai kebutuhan
def train_choosing_feature(dataset,file_header,file_data):
    """
    Param :
    dataset = pandas dataframe
        dataframe dari hasil pembacaan input dataset
    file_header = string
        Path dan nama file menyimpan data header
    file_data = string
        Path dan nama file menyimpan dataframe baru
    """
    array = dataset.values
    X = array[:,1:-1]
    Y = array[:,-1]
    feature_col = list(dataset.columns.values)
    selector = SelectKBest(f_classif,k=12).fit(X, Y)
    mask = selector.get_support()
    newF = ['shop_id']
    for bool,feature in zip(mask,feature_col):
        if bool:
            newF.append(feature)
    newDF = []
    for x in range(len(newF)):
        newDF.append(dataset[newF[x]])

    newDF.append(dataset['approval'])
    newDF = pd.concat(newDF,axis=1)
    newF.append('approval')
    selected_feature(newDF,newF,file_header,file_data) # Mengirimkan ke fungsi lain

# Fungsi untuk menyimpan dataset baru dan nama header dari fitur seleksi
def selected_feature(newDF,newF,file_header,file_data):
    """
    Param :
    newDF = pandas dataframe
        dataframe dari hasil seleksi fitur
    newF = list
        list berisikan nama fitur hasil seleksi
    file_header = string
        Path dan nama file menyimpan data header
    file_data = string
        Path dan nama file menyimpan dataframe baru
    """
    # Menyimpan dataset dengan fitur pilihan
    newDF.to_csv(file_data , index = False)
    # Menyimpan nama header fitur pilihan
    for i in range(len(newF)):
        with open(file_header,"ab") as txtfile:
            writer = unicodecsv.writer(txtfile, delimiter = ',', quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerow([newF[i]])
    txtfile.close()

# Fungsi untuk mengubah dataset test menyesuaikan dengan fitur terpilih
# Fungsi ini juga menyimpan dataset test yang baru 
def test_choosing_feature(dataset,file_header,file_data):
    """
    Param :
    dataset = pandas dataframe
        dataframe dari hasil pembacaan input dataset
    file_header = string
        Path dan nama file penyimpan data header
    file_data = string
        Path dan nama file menyimpan dataframe baru
    """
    header_feature = []
    # Membaca data file header
    with open(file_header,'r+') as txtfile:
        reader = csv.reader(txtfile)
        for item in reader:
            header_feature.append(item)
    newDF = []
    for i in range(len(header_feature)):
        newDF.append(dataset[header_feature[i]])
    newDF = pd.concat(newDF,axis=1)
    # Menyimpan data set baru
    newDF.to_csv(file_data , index = False)
    new_data = load_dataset(file_data)
    return new_data # dataset baru dengan fitur pilihan

# Fungsi learning yang menggunakan metode LR dan penyimpanan model
def learning(dataset,model):
    """
    Param :
    dataset = pandas dataframe
        dataframe dari hasil pembacaan input dataset
    model = string
        Path dan nama model untuk disimpan
    """
    array = dataset.values
    X = array[:,1:-1]
    Y = array[:,-1]
    seed = 7
    lr = LogisticRegression(class_weight="balanced",random_state=seed,C=0.9)
    lr.fit(X, Y)
    save_model(lr,model) # Penyimpanan model menggunakan fungsi lain
    evaluating(array,model) # Evaluasi hasil model

# Fungsi evaluasi dari model hasil learning
def evaluating(array,filename):
    """
    Param :
    array = list
        list dari values dari dataframe
    filename = string
        Path dan nama model yang disimpan
    """
    print("Evaluation")
    X = array[:,1:-1]
    Y = array[:,-1] #label yang sebenarnya
    loaded_model = load_model(filename)
    Y_predicted = loaded_model.predict(X)
    ##### metrics evaluasi #####
    # hitung score, bandingkan hasil label prediksi, dengan label sesungguhnya di testing data
    # tampilkan nilai accuracy
    accuracy = accuracy_score(Y, Y_predicted)
    print('Accuracy : ', accuracy)
    # metric lebih detail: precision, recall, f1
    report = classification_report(Y, Y_predicted)
    print ('\nPrecision, Recall, f1:')
    print (report)

    # confusion matrix
    print ('\nConfusion Matrix:')
    print (confusion_matrix(Y, Y_predicted))

# Fungsi testing yang digunakan untuk melakukan test terhadap dataset baru
def testing(dataset,model,filename):
    """
    Param :
    dataset = pandas dataframe
        dataframe dari hasil pembacaan input dataset
    model = string
        Path dan nama model yang disimpan
    filename = string
        Path dan nama file menyimpan dataframe baru hasil test
    """
    array = dataset.values
    X = array[:,1:13]
    shop_id = array[:,0]
    loaded_model = load_model(model)
    Y = loaded_model.predict(X)
    test = pd.DataFrame( { 'shop_id': shop_id , 'approval_pred': Y} )
    test.to_csv( filename , index = False )
    return print("Success")

# Fungsi untuk save learned model ke sebuah file
def save_model(model, filename):
    """
    Param :
    model = object 
        Model dari hasil machine learning
    filename = string
        Path dan nama file menyimpan model hasil machine learning
    """
    pickle.dump(model, open(filename, 'wb'))
    return print("Saving Model Succes\n")

# Fungsi untuk load kembali model yang sudah di-save pada suatu file
def load_model(filename):
    """
    Param :
    filename = string
        Path dan nama model disimpan
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

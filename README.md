# Prediksi_cuaca_dengan_JST

PREDIKSI CUACA DENGAN ARTIFICIAL NEURAL NETWORK
STEP BY STEP :

Upload dataset
In [2]:
import pandas as pd
from google.colab import files

files = files.upload()
Saving day.csv to day.csv
Pastikan dataset bisa terbaca
In [3]:
import io

data = pd.read_csv(io.BytesIO(files['day.csv']))
data1 = data
data2 = data1
data3 = data2

print(data)
     instant      dteday  season  yr  ...  windspeed  casual  registered   cnt
0          1  2011-01-01       1   0  ...   0.160446     331         654   985
1          2  2011-01-02       1   0  ...   0.248539     131         670   801
2          3  2011-01-03       1   0  ...   0.248309     120        1229  1349
3          4  2011-01-04       1   0  ...   0.160296     108        1454  1562
4          5  2011-01-05       1   0  ...   0.186900      82        1518  1600
..       ...         ...     ...  ..  ...        ...     ...         ...   ...
726      727  2012-12-27       1   1  ...   0.350133     247        1867  2114
727      728  2012-12-28       1   1  ...   0.155471     644        2451  3095
728      729  2012-12-29       1   1  ...   0.124383     159        1182  1341
729      730  2012-12-30       1   1  ...   0.350754     364        1432  1796
730      731  2012-12-31       1   1  ...   0.154846     439        2290  2729

[731 rows x 16 columns]
Data sudah berhasil dibaca, setelah itu pilih data yang akan dipakai. untuk memprediksi cuaca kita membutuhkan data

Temperature
Kelembapan
Kecepatan angin
dan data targer yaitu

cuaca
In [4]:
dataset  = data[['temp','hum','windspeed','weathersit']]
print('----------------------------DATASET--------------------------------------------')
print(dataset)
----------------------------DATASET--------------------------------------------
         temp       hum  windspeed  weathersit
0    0.344167  0.805833   0.160446           2
1    0.363478  0.696087   0.248539           2
2    0.196364  0.437273   0.248309           1
3    0.200000  0.590435   0.160296           1
4    0.226957  0.436957   0.186900           1
..        ...       ...        ...         ...
726  0.254167  0.652917   0.350133           2
727  0.253333  0.590000   0.155471           2
728  0.253333  0.752917   0.124383           2
729  0.255833  0.483333   0.350754           1
730  0.215833  0.577500   0.154846           2

[731 rows x 4 columns]
Transformasi data ke array matrix
In [5]:
import numpy as np

temperature = np.array(data['temp'])
Kelembapan = np.array(data1['hum'])
kecepatan_angin = np.array(data2['windspeed'])
Cuaca = np.array(data3['weathersit'])


matrix = [temperature,Kelembapan,kecepatan_angin,Cuaca]
print(np.array(matrix,ndmin=2).T)
[[0.344167 0.805833 0.160446 2.      ]
 [0.363478 0.696087 0.248539 2.      ]
 [0.196364 0.437273 0.248309 1.      ]
 ...
 [0.253333 0.752917 0.124383 2.      ]
 [0.255833 0.483333 0.350754 1.      ]
 [0.215833 0.5775   0.154846 2.      ]]
inisialisasi komponen ANN a. Learning rate b. bias c. bobot
In [0]:
import random
import os

learning_rate = 1
bias = 1
weights = [random.random(),random.random(),random.random(),random.random()]
4 bobot untuk 3 input layer dan 4 hidden layer

buat fungsi Perceptron
In [0]:
def Perceptron(input1,input2,input3,output):
  outputP = input1*weights[0]+input2*weights[1]+input3*weights[2]+bias*weights[3]
  outputP = 1/(1+np.exp(-outputP)) # Sigmoid
  
  error = output - outputP
  weights[0]+= error*input1*learning_rate
  weights[1]+= error*input2*learning_rate
  weights[2]+= error*input3*learning_rate
  weights[3]+= error*bias*learning_rate
buat perulangan untuk melatih data
In [0]:
for i in range(50):
  Perceptron(temperature,Kelembapan,kecepatan_angin,Cuaca)
  Perceptron(temperature,Kelembapan,kecepatan_angin,Cuaca)
  Perceptron(temperature,Kelembapan,kecepatan_angin,Cuaca)
  Perceptron(temperature,Kelembapan,kecepatan_angin,Cuaca)
  Perceptron(temperature,Kelembapan,kecepatan_angin,Cuaca)
buat inputan
In [17]:
x = float(input("Masukan Nilai Temperature     :"))
y = float(input("Masukan Nilai Kelembapan      :"))
z = float(input("Masukan Nilai Kecepatan Angin :"))

outputP = x*weights[0]+y*weights[1]*z*weights[2]+bias*weights[3]
outputP = 1/(1+np.exp(-outputP))

if outputP[0] <= 1:
  outputP = 1
  outputP =str('Cerah') 
elif outputP[0] ==2:
  outputP = 2
  outputP =str('Berawan')
elif outputP[0] ==3:
  outputP = 3
  outputP =str('Gerimis')
else:
  outputP = 4
  outputP =str('Hujan')

print("")
print("Temperature :",x,"Kelembapan :",y,"WindSpeed :",z,"Prediksi :",outputP)
Masukan Nilai Temperature     :0.8976
Masukan Nilai Kelembapan      :0.8765
Masukan Nilai Kecepatan Angin :0.6557

Temperature : 0.8976 Kelembapan : 0.8765 WindSpeed : 0.6557 Prediksi : Cerah


Terimakasih... itulah program sederhana yang dibuat dengan ANN

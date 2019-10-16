import numpy as np

def readData():
    data = np.genfromtxt('./data/admission_predict.csv', delimiter= ',')
    return data[1:, 1:]






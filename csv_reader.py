import os
import sys
import random
import numpy as np
import csv
import pickle
import pandas as pd

FILE_REPOSITORY = './pulsar_dataBase/'
FILE_NAME = 'pulsar_stars.csv'

with open(FILE_REPOSITORY+FILE_NAME , newline='') as csvfile :
    CSVreader = csv.reader(csvfile, delimiter=',')
    
    data_base = pd.DataFrame(CSVreader)


pickle.dump( data_base, open( "save_data_pulsar.p", "wb" ) )


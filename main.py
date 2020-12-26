import os
os.chdir('/Users/U1189873/Documents/Marsh/Projects/NLP/LanguageModels/CharElectra/MyELECTRA')

import tensorflow as tf
import pandas as pd
from model import MyELECTRA
from utilities.utils import prepare_text_training


corpus = ['hello my name is laurent and i am a data scientist trying to understand the rule of thumb when blabla'.split(),
        'i am a data scientist on the weekdays'.split()]

parameters = {
    'd_model' : 256,
    'dff' : 1024,
    'pe_input' : 150,
    'num_layers' : 12,
    'd_embeddings' : 64,
    'filters' : {1: 16, 2 : 16, 3: 32, 4: 64, 5: 128, 6: 256, 7: 512},
    'num_highway_layers' : 2,
    'fitted' : False,
    'hs' : True
}

path = os.getcwd()
self = MyELECTRA(parameters, path_model = os.path.join(path, 'model'))
self.fit(corpus, batch_size = 32, epochs = 1, masking_rate = .15, min_count = 1)


import time

path_data = r"/Users/laurentthanwerdas/Documents/Documents/Etudes/NY/Personal/PROJECTS/Deep_Embedded_Clustering/severeinjury.csv"
data = pd.read_csv(path_data, encoding = 'latin9').head(1000)
corpus, _ = prepare_text_training(data['text'])
corpus = [c for c in corpus if len(c) < 126]

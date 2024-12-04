from word2vec import Word2Vec
import pandas as pd
import time
import shutil

try:
    shutil.rmtree('.word2vec')
except FileNotFoundError:
    pass

df = pd.read_csv('LLM.csv')

start_time = time.time()
Word2Vec(df)
print(f'First load time: {(time.time() - start_time):.2f} seconds')

start_time = time.time()
Word2Vec(df)
print(f'Second load time: {(time.time() - start_time):.2f} seconds')

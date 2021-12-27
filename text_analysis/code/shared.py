import gensim
import time


model_path = '../model/wiki-news-300d-1M-subword.vec'
print('Loading model')
start_time = time.time()
model = gensim.models.keyedvectors.load_word2vec_format(model_path)
print(f'Model loaded in {round(time.time() - start_time, 2)} seconds\n')

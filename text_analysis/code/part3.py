import gensim
import json
import os
import time

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.parsing.preprocessing import *
from typing import List


class ProgressLogger(CallbackAny2Vec):

    def __init__(self, total_epochs):
        self.epoch = 0
        self.total_epochs = total_epochs
        self.start_time = time.time()

    def on_epoch_begin(self, model):
        self.epoch += 1
        self.start_time = time.time()

    def on_epoch_end(self, model):
        print(f'{self.epoch}/{self.total_epochs}: Epoch took {round(time.time() - self.start_time, 2)} seconds')


def get_texts_from_dataset() -> List[str]:
    print('Loading dataset')
    start_time = time.time()
    texts = []
    files_names: List[str] = os.listdir('./german_dataset')
    for file_name in files_names:
        with open(f'./german_dataset/{file_name}') as file:
            tweets = json.load(file)
            for tweet in tweets:
                if 'text' in tweet:
                    texts.append(tweet['text'])
    print(f'Dataset loaded in {round(time.time() - start_time, 2)} seconds\n')
    print_size_of_texts(texts)
    return texts


def print_size_of_texts(texts: List[str]):
    size = 0
    for text in texts:
        size += len(text.encode('utf-8'))
    print(f'Dataset size: {round(size / (1024 * 1024 * 1024), 2)}GB')


def train_model(texts: List[str]) -> Word2Vec:
    print('Training model')
    start_time = time.time()
    filters = [strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, strip_short]
    texts_tokenized = [preprocess_string(text.lower(), filters) for text in texts]

    epochs = 10
    german_model = gensim.models.Word2Vec(sentences=texts_tokenized, vector_size=100, window=5, min_count=1, epochs=epochs,
                                          workers=12, callbacks=[ProgressLogger(epochs)])
    print(f'Model trained in {round(time.time() - start_time, 2)} seconds\n')
    return german_model


def save_model(model: Word2Vec):
    print('Saving model')
    start_time = time.time()
    model.save('../model/german-tweets.vec')
    print(f'Model saved in {round(time.time() - start_time, 2)} seconds\n')


def main():
    texts = get_texts_from_dataset()
    model = train_model(texts)
    save_model(model)


if __name__ == '__main__':
    main()

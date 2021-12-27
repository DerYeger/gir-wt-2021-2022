import gensim
import multiprocessing
import os
import time

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.parsing.preprocessing import *
from shared import get_three_most_similar
from typing import List

_dataset_dir = './german/dataset'
_model_file_path = './german/model.vec'


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
    files_names: List[str] = os.listdir(_dataset_dir)
    for file_name in files_names:
        with open(f'{_dataset_dir}/{file_name}', encoding='utf-8') as file:
            lines = file.read().split('\n')
            texts.extend(lines)
    print(f'Dataset loaded in {round(time.time() - start_time, 2)} seconds')
    print_size_of_texts(texts)
    return texts


def print_size_of_texts(texts: List[str]):
    size = 0
    for text in texts:
        size += len(text.encode('utf-8'))
    print(f'Dataset size: {round(size / (1024 * 1024 * 1024), 2)}GB\n')


def train_model(texts: List[str]) -> KeyedVectors:
    print('Training model')
    num_threads = multiprocessing.cpu_count()
    num_workers = 1 if num_threads <= 1 else num_threads - 1
    print(f'Using {num_workers} of {multiprocessing.cpu_count()} available threads')
    start_time = time.time()

    epochs = 10
    texts_tokenized = process_texts(texts)
    german_model = gensim.models.Word2Vec(sentences=texts_tokenized, vector_size=100, window=5, min_count=1,
                                          epochs=epochs, workers=num_workers, callbacks=[ProgressLogger(epochs)])
    print(f'Model trained in {round(time.time() - start_time, 2)} seconds\n')
    return german_model.wv


def save_model(model: KeyedVectors):
    print('Saving model')
    start_time = time.time()
    model.save_word2vec_format(_model_file_path)
    print(f'Model saved in {round(time.time() - start_time, 2)} seconds\n')


def load_model() -> KeyedVectors:
    print('Loading model')
    start_time = time.time()
    model = gensim.models.keyedvectors.load_word2vec_format(_model_file_path)
    print(f'Model loaded in {round(time.time() - start_time, 2)} seconds\n')
    return model


def get_model() -> KeyedVectors:
    if os.path.exists(_model_file_path):
        return load_model()
    else:
        texts = get_texts_from_dataset()
        model = train_model(texts)
        save_model(model)
        return model


def process_texts(texts: List[str]) -> List[str]:
    filters = [strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, strip_short]
    return [preprocess_string(text.lower(), filters) for text in texts]


def main():
    model = get_model()
    for [word] in process_texts(['Deutschland', 'Politik', 'Kanzler']):
        get_three_most_similar(word, model)


if __name__ == '__main__':
    main()

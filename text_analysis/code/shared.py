import gensim
import time

from gensim.models import KeyedVectors
from typing import Tuple

_english_model_path = '../model/wiki-news-300d-1M-subword.vec'
_english_model = None


def get_english_model() -> KeyedVectors:
    global _english_model
    if _english_model is None:
        print('First access to English model')
        print('Loading model')
        start_time = time.time()
        _english_model = gensim.models.keyedvectors.load_word2vec_format(_english_model_path)
        print(f'Model loaded in {round(time.time() - start_time, 2)} seconds\n')
    return _english_model


def get_three_most_similar(word: str, model: KeyedVectors):
    if model.has_index_for(word):
        print(f'Words similar to "{word}":')
        results = model.most_similar(word, topn=3)
        for i, (otherWord, score) in enumerate(results):
            print(f'{i + 1}. {otherWord} ({score})')
        print()
    else:
        print(f'Model does not contain word "{word}"')


def get_cosine_similarity(pair: Tuple[str, str], model: KeyedVectors):
    result = model.similarity(pair[0], pair[1])
    print(f'cos_sim({pair[0]}, {pair[1]}) = {result}\n')

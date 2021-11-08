import numpy as np

from inverted_index import InvertedIndex

scoring_modes = ['bm25', 'tf-idf']


def bm25(index: InvertedIndex, query_tokens: [str]):
    article_scores = {}
    for token in query_tokens:
        token_idf = _idf(index, token)
        for (article_id, frequency) in index.get_entries_for_token(token):
            id_string = str(article_id)
            if id_string not in article_scores:
                article_scores[id_string] = 0
            k = (1.2 + 2) / 2
            b = 0.75
            word_count = index.get_article_by_id(id_string)[3]
            nominator = frequency * (k + 1)
            denominator = frequency + k * (1 - b + (b * (word_count / index.get_average_word_count())))
            article_scores[id_string] += token_idf * (nominator / denominator)
            article_scores[id_string] += token_idf * frequency
    return article_scores


def tf_idf(index: InvertedIndex, query_tokens: [str]):
    article_scores = {}
    for token in query_tokens:
        token_idf = _idf(index, token)
        for (article_id, frequency) in index.get_entries_for_token(token):
            id_string = str(article_id)
            if id_string not in article_scores:
                article_scores[id_string] = 0
            article_scores[id_string] += token_idf * frequency
    return article_scores


def _idf(index: InvertedIndex, token: str):
    article_count = index.get_article_count()
    n_qi = len(index.get_entries_for_token(token))
    nominator = article_count - n_qi + 0.5
    denominator = n_qi + 0.5
    return np.log((nominator / denominator) + 1)

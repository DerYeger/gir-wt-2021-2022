import numpy as np

from inverted_index import InvertedIndex
from typing import Dict, List

bm25_name = 'BM25'
tf_idf_name = 'TF-IDF'

scoring_modes: List[str] = [bm25_name, tf_idf_name]


def score(index: InvertedIndex, query_tokens: List[str], eval_type: str) -> Dict[int, float]:
    if eval_type is bm25_name:
        return _bm25(index, query_tokens)
    elif eval_type is tf_idf_name:
        return _tf_idf(index, query_tokens)
    else:
        return {}


def _bm25(index: InvertedIndex, query_tokens: List[str]) -> Dict[int, float]:
    article_scores: Dict[int, float] = {}
    k = (1.2 + 2) / 2
    b = 0.75
    for token in query_tokens:
        token_idf: float = _idf(index, token)
        for (article_id, frequency) in index.get_entries_for_token(token):
            if article_id not in article_scores:
                article_scores[article_id] = 0
            word_count = index.get_article_by_id(article_id)[2]
            nominator: float = frequency * (k + 1)
            denominator: float = frequency + k * (1 - b + (b * (word_count / index.get_average_word_count())))
            article_scores[article_id] += token_idf * (nominator / denominator)
    return article_scores


def _tf_idf(index: InvertedIndex, query_tokens: List[str]) -> Dict[int, float]:
    article_scores: Dict[int, float] = {}
    for token in query_tokens:
        token_idf: float = _idf(index, token)
        for (article_id, frequency) in index.get_entries_for_token(token):
            article_tf: float = np.log(1 + frequency)
            if article_id not in article_scores:
                article_scores[article_id] = 0
            article_scores[article_id] += token_idf * article_tf
    return article_scores


def _idf(index: InvertedIndex, token: str) -> float:
    article_count: int = index.get_article_count()
    return np.log(article_count / (len(index.get_entries_for_token(token)) + 1))

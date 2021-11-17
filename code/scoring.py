import numpy as np

from inverted_index import InvertedIndex

bm25_name = 'BM25'
tf_idf_name = 'TF-IDF'

scoring_modes: list[str] = [bm25_name, tf_idf_name]


def score(index: InvertedIndex, query_tokens: list[str], eval_type: str) -> dict[str, float]:
    if eval_type is bm25_name:
        return _bm25(index, query_tokens)
    elif eval_type is tf_idf_name:
        return _tf_idf(index, query_tokens)
    else:
        return {}


def _bm25(index: InvertedIndex, query_tokens: list[str]) -> dict[str, float]:
    article_scores: dict[str, float] = {}
    k = (1.2 + 2) / 2
    b = 0.75
    for token in query_tokens:
        token_idf: float = _idf(index, token)
        for (article_id, frequency) in index.get_entries_for_token(token):
            id_string: str = str(article_id)
            if id_string not in article_scores:
                article_scores[id_string] = 0
            word_count: int = index.get_article_by_id(id_string)[2]
            nominator: float = frequency * (k + 1)
            denominator: float = frequency + k * (1 - b + (b * (word_count / index.get_average_word_count())))
            article_scores[id_string] += token_idf * (nominator / denominator)
    return article_scores


def _tf_idf(index: InvertedIndex, query_tokens: list[str]) -> dict[str, float]:
    article_scores: dict[str, float] = {}
    for token in query_tokens:
        token_idf: float = _idf(index, token)
        for (article_id, frequency) in index.get_entries_for_token(token):
            id_string: str = str(article_id)
            article_tf: float = np.log(1 + frequency)
            if id_string not in article_scores:
                article_scores[id_string] = 0
            article_scores[id_string] += token_idf * article_tf
    return article_scores


def _idf(index: InvertedIndex, token: str) -> float:
    article_count: int = index.get_article_count()
    return np.log(article_count / (len(index.get_entries_for_token(token)) + 1))

from inverted_index import InvertedIndex
from scoring import bm25, tf_idf
from tokenizer import tokenize


def query(index: InvertedIndex, query_string: str, eval_type: str):
    query_tokens = tokenize(query_string)
    scores = tf_idf(index, query_tokens) if eval_type == 'tf-idf' else bm25(index, query_tokens)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

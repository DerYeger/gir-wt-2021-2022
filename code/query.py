from inverted_index import InvertedIndex
from scoring import bm25, tf_idf
from tokenizer import tokenize
from utils import highlight


def query(index: InvertedIndex, query_string: str, eval_type: str, silent=False):
    query_tokens = tokenize(query_string)
    scores = tf_idf(index, query_tokens) if eval_type == 'tf-idf' else bm25(index, query_tokens)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    for rank, (article_id, article_score) in enumerate(sorted_results):
        article_title = index.get_article_by_id(str(article_id))[0]
        if not silent:
            print(f'{highlight(f"#{rank + 1}")} is article {highlight(article_id)} with score {highlight(article_score)} and title {highlight(article_title)}')
    return sorted_results

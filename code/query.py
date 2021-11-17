from inverted_index import InvertedIndex
from scoring import score
from tokenizer import tokenize


def query(index: InvertedIndex, query_string: str, eval_type: str) -> list[(str, float)]:
    query_tokens: list[str] = tokenize(query_string)
    scores: dict[int, float] = score(index, query_tokens, eval_type)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

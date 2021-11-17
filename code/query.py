from inverted_index import InvertedIndex
from scoring import score
from tokenizer import tokenize
from typing import Dict, List, Tuple


def query(index: InvertedIndex, query_string: str, eval_type: str) -> List[Tuple[int, float]]:
    query_tokens: List[str] = tokenize(query_string)
    scores: Dict[int, float] = score(index, query_tokens, eval_type)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

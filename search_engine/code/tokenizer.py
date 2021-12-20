import re

from nltk import download
from nltk.stem import SnowballStemmer, StemmerI
from nltk.corpus import stopwords
from typing import Callable, Dict, List

download('stopwords', quiet=True)
_stop_words: List[str] = stopwords.words('english')

_stemmer: StemmerI = SnowballStemmer('english')
_stemmer_cache: Dict[str, str] = {}

_punctuation_pattern = re.compile(r'[.,()\[\]{}:;\n\t\s \\|/?!]')
_non_word_pattern = re.compile(r'[\Wˆ_]')


def tokenize(text: str) -> List[str]:
    """
    :param text: a text string
    :return: an array of tokens with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    raw_tokens: List[str] = _punctuation_pattern.split(text.lower().strip())
    processed_tokens = _flat_map(_process_token, raw_tokens)
    filtered_tokens = filter(_token_filter, processed_tokens)
    stemmed_tokens = map(_stem, filtered_tokens)
    return list(stemmed_tokens)


def _flat_map(transform: Callable[[str], List[str]], tokens: List[str]) -> List[str]:
    return [result for token in tokens for result in transform(token)]


def _process_token(word: str) -> [str]:
    parts = word.split('-')
    if len(parts) > 1:
        parts.append(word)
    return [_non_word_pattern.sub('', part) for part in parts]


def _stem(word: str) -> str:
    if word not in _stemmer_cache:
        _stemmer_cache[word] = _stemmer.stem(word)
    return _stemmer_cache[word]


def _token_filter(token: str) -> bool:
    return token not in _stop_words and (len(token) > 1 or token.isnumeric())

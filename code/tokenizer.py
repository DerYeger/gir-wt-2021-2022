import re

from nltk import download
from nltk.stem import LancasterStemmer, StemmerI
from nltk.corpus import stopwords
from typing import List

download('stopwords', quiet=True)
_stop_words: List[str] = stopwords.words('english')

_stemmer: StemmerI = LancasterStemmer()


def tokenize(text: str) -> List[str]:
    """
    :param text: a text string
    :return: an array of tokens with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    # replace non-word chars with spaces and split at spaces
    raw_tokens: List[str] = re.sub(r'[\Wˆ_]+', ' ', text).strip().lower().split()
    filtered_tokens: filter = filter(lambda t: t not in _stop_words, raw_tokens)
    stemmed_tokens: map = map(_stemmer.stem, filtered_tokens)
    # normalized_tokens = map(replace_special_letters, stemmed_tokens)
    return list(stemmed_tokens)


def _replace_special_letters(word: str) -> str:
    return word \
        .replace('ä', 'a').replace('â', 'a').replace('á', 'a').replace('à', 'a').replace('ã', 'a').replace('ȧ', 'a') \
        .replace('ç', 'c') \
        .replace('ë', 'e').replace('ê', 'e').replace('é', 'e').replace('è', 'e') \
        .replace('ï', 'i').replace('î', 'i').replace('í', 'i').replace('ì', 'i') \
        .replace('ñ', 'n') \
        .replace('ö', 'o').replace('ô', 'o').replace('ó', 'o').replace('ò', 'o').replace('õ', 'o') \
        .replace('ß', 'ss') \
        .replace('ü', 'u').replace('û', 'u').replace('ú', 'u').replace('ù', 'u') \
        .replace('ÿ', 'y')

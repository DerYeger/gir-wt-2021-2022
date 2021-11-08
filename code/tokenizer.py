import re

from nltk import download
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

download('stopwords', quiet=True)
_stop_words = stopwords.words('english')

_stemmer = PorterStemmer()


def tokenize(text: str) -> [str]:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    # remove double spaces, tabs and special chars
    clean_text: str = ' '.join(re.sub(r'[,.;:?/(){}\[\]\-‑|_+=\'’`"”“!@#$%^&*<>]', ' ', text).strip().split())
    # split a punctuations and spaces etc.
    raw_tokens: [str] = filter(None, re.split(r'[\s\n]', clean_text))
    lowercase_tokens = map(lambda t: t.lower(), raw_tokens)
    filtered_tokens = filter(lambda t: t not in _stop_words, lowercase_tokens)
    stemmed_tokens = map(_stem, filtered_tokens)
    # normalized_tokens = map(replace_special_letters, stemmed_tokens)
    return stemmed_tokens


def _stem(word: str) -> str:
    return _stemmer.stem(word)


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

"""
This file contains your code to create the inverted index.
Besides implementing and using the predefined tokenization function (text2tokens),
there are no restrictions in how you organize this file.
"""
import re
from nltk.stem import PorterStemmer
import os
from bs4 import BeautifulSoup
import time

inverted_index_table = dict()
documents_table = dict({1: ['sdf', 234]})

actual_dir: str = './wiki_files/dataset/articles/'
test_dir: str = './wiki_files/test/'
current_dir: str = test_dir


def load_wiki_files():
    file_entry: str
    for file_entry in os.listdir(current_dir):
        print(file_entry)
        with open(current_dir + file_entry, encoding='utf8') as file:
            eval_wiki_data(file)


def eval_wiki_data(file):
    soup = BeautifulSoup(file.read(), 'html.parser')
    for article in soup.find_all('article'):
        article_id = article.find('id').string  # get article id
        article_body = article.find('bdy')  # get article body
        # get content of article body or empty string if body does not exist
        article_content = '' if article_body is None else article_body.string
        article_tokens = tokenization(article_content)
        print('\nArticle {}: {}\n'.format(article_id, article_tokens))
    pass


def text2tokens(text: str) -> [str]:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    tokens: [str] = tokenization(text)
    print('tokens:', tokens)
    return tokens


# https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/
stopWords = {}
with open('./tool_data/stop_words.txt') as stop_words:
    stop_words = {w for w in stop_words.read().split(',')}


def tokenization(text: str) -> [str]:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """

    clean_text: str = ''.join(text.strip().split())  # remove double spaces or tabs
    raw_tokens = re.split('[\s]*[\s.,;:]+[\s]*', clean_text)  # split a punctuations and spaces etc.

    # print(clean_text)
    # print(raw_tokens)

    tokens: [str] = []
    for w in raw_tokens:
        if len(w) <= 0:
            print('ERROR raw token with length 0 found')
        w = w.lower().replace('-', '').replace('\'', '')
        if w in stopWords:
            continue
        w = stemming(w)
        if w in stopWords:
            continue
        tokens.append(w)
    return tokens


def stemming(word: str):
    """
    :param word: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    # https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

    # create an object of class PorterStemmer
    porter = PorterStemmer()
    word = porter.stem(word)

    # lancaster = LancasterStemmer()
    # word = lancaster.stem(word)

    return word


if __name__ == '__main__':
    start_time = time.time()
    load_wiki_files()
    print("--- %s seconds ---" % (time.time() - start_time))
    text2tokens("([hello]   {world}))")
    text2tokens(
        'cats houses complementations sadf efw the a is this weasdf. fewfsdf .ssdfsssssssss.\nfdsfew.   fewfds    '
        '.asdf, fdsdfs. To, By this is a not ')

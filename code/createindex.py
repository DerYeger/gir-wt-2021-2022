"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import os
from bs4 import BeautifulSoup

inverted_index_table = dict()
documents_table = dict({1: ["sdf", 234]})




def load_wiki_files():
    for file_entry in os.listdir("./wiki_files/test"):
        if file_entry.endswith(".txt"):
            print(file_entry)
            with open('./wiki_files/test/' + file_entry) as file:
                eval_wiki_data(file)


def eval_wiki_data(file):
    soup = BeautifulSoup(file.read(), 'html.parser')
    for id in soup.find_all("id"):
        print(id)
    pass


def text2tokens(text) -> str:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    tokens = tokenization(text)
    print('tokens:', tokens)
    return tokens


# https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/
stopWords = {}
with open('./tool_data/stop_words.txt') as stop_words:
    stopWords = {w for w in stop_words.read().split(',')}


def tokenization(text: str) -> []:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """

    clean_text = ' '.join(text.strip().split())  # remove double spaces or tabs
    raw_tokens = re.split('[\s]*[\s.,;:]+[\s]*', clean_text)  # split a punctuations and spaces etc.
    raw_tokens.pop()  # last entry is empty and therefore removed

    # print(clean_text)
    # print(raw_tokens)

    tokens = []
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
    :param text: a text string
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
    load_wiki_files()
    text2tokens(
        "cats houses complementations sadf efw the a is this weasdf. fewfsdf .ssdfsssssssss.\nfdsfew.   fewfds    .asdf, fdsdfs. To, By this is a not ")

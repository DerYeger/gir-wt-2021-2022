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

inverted_index_table = {}
documents_table = {}

actual_dir: str = './wiki_files/dataset/articles/'
test_dir: str = './wiki_files/test/'
current_dir: str = actual_dir

max_files: int = 1

# https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/
stopWords = {}
with open('./tool_data/stop_words.txt') as stop_words:
    stop_words = {w for w in stop_words.read().split(',')}

stemmer = PorterStemmer()


def stem(word: str) -> str:
    return stemmer.stem(word)


def load_wiki_files():
    file_entry: str
    files_read = 0
    for file_entry in os.listdir(current_dir):
        if max_files == files_read:
            return
        print(file_entry)
        with open(current_dir + file_entry, encoding='utf8') as file:
            eval_wiki_data(file)

        curr_time = time.time()
        if files_read > 0:
            print("--- %s seconds for file number %d---" % (curr_time - last_time, files_read))
        last_time = time.time()
        files_read += 1


def eval_wiki_data(file):
    soup = BeautifulSoup(file.read(), 'html.parser')
    for article in soup.find_all('article'):
        article_id: str = article.find('id').string  # get article id
        article_title: str = article.find('title').string  # get article id
        article_body = article.find('bdy')  # get article body
        # get content of article body or empty string if body does not exist
        article_content: str = '' if article_body is None else article_body.string
        article_tokens: [str] = [*tokenization(article_content)]
        # print('\nArticle {}: {} tokens\n'.format(article_id, article_tokens))
        list(map(lambda token: insert_index(article_id, token), article_tokens))
        documents_table[article_id] = [article_title, file.name, soup.index(article)]
        pass


def text2tokens(text: str) -> [str]:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    tokens: [str] = tokenization(text)
    print('tokens:', tokens)
    return tokens


def tokenization(text: str) -> [str]:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    # remove double spaces, tabs and special chars
    clean_text: str = ' '.join(re.sub(r'[-(){}\[\]\'"]', '', text).strip().split())
    # split a punctuations and spaces etc.
    raw_tokens: [str] = list(filter(None, re.split(r'[\s]*[\s.,;:\n]+[\s]*', clean_text)))
    lowercase_tokens = map(lambda token: token.lower(), raw_tokens)
    stemmed_tokens = map(stem, lowercase_tokens)
    filtered_tokens = filter(lambda token: token not in stop_words, stemmed_tokens)
    return list(filtered_tokens)


if __name__ == '__main__':
    start_time = time.time()
    load_wiki_files()
    print("--- %s seconds ---" % (time.time() - start_time))
    text2tokens("([house]   {houses}))")
    text2tokens(
        'cats houses complementations sadf efw the a is this weasdf. fewfsdf .ssdfsssssssss.\nfdsfew.   fewfds    '
        '.asdf, fdsdfs. To, By this is a not ')
    print(inverted_index_table)
    print(documents_table)

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
import numpy as np
import ast
import codecs


# https://dev.to/turbaszek/flat-map-in-python-3g98
def flat_map(f, xs):
    return [y for ys in xs for y in f(ys)]


# Settings
encoding = 'utf_16'

inverted_index_table = {}
article_table = {}  # {article_id: [title, path, offset, token_occurrences: dict, word count]}
avg_dl = 0  # average document length in words

actual_dir: str = './wiki_files/dataset/articles/'
test_dir: str = './wiki_files/test/'
current_dir: str = actual_dir

max_files: int = 1

# https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/
stopWords = {}
with open('./tool_data/stop_words.txt') as stop_words:
    stop_words = {w for w in stop_words.read().split(',')}

stemmer = PorterStemmer()


def save_tables():
    with codecs.open('./tables/inverted_index_table.txt', 'w+', encoding) as f:
        f.write(str(inverted_index_table))
    with codecs.open('./tables/article_table.txt', 'w+', encoding) as f:
        f.write(str(article_table))
    with codecs.open('./tables/avg_dl.txt', 'w+', encoding) as f:
        f.write(str(avg_dl))


def load_tables() -> bool:
    global inverted_index_table, article_table, avg_dl

    if not os.path.exists('./tables/inverted_index_table.txt') or not os.path.exists(
            './tables/article_table.txt') or not os.path.exists('./tables/avg_dl.txt'):
        return False

    with codecs.open('./tables/inverted_index_table.txt', 'r', encoding) as f:
        table = f.read()
        inverted_index_table = set() if table == str(set()) else ast.literal_eval(table)
    with codecs.open('./tables/article_table.txt', 'r', encoding) as f:
        table = f.read()
        article_table = set() if table == str(set()) else ast.literal_eval(table)
    with codecs.open('./tables/avg_dl.txt', 'r', encoding) as f:
        value = f.read()
        avg_dl = ast.literal_eval(value)
    return True


def stem(word: str) -> str:
    return stemmer.stem(word)


def load_wiki_files():
    file_entry: str
    files_read = 0
    last_time = time.time()
    for file_entry in os.listdir(current_dir):
        if max_files == files_read:
            return
        print(file_entry)
        with open(current_dir + file_entry, encoding='utf8') as file:
            eval_wiki_data(file)

        curr_time = time.time()
        print("--- %s seconds for file number %d---" % (curr_time - last_time, files_read + 1))
        last_time = time.time()
        files_read += 1


def insert_index(article_id: str, token: str, token_occurrences: dict):
    if token in inverted_index_table:
        inverted_index_table[token].add(article_id)
    else:
        inverted_index_table[token] = {article_id}

    if token in token_occurrences:
        token_occurrences[token] += 1
    else:
        token_occurrences[token] = 1


def eval_wiki_data(file):
    global avg_dl
    soup = BeautifulSoup(file.read(), 'html.parser')
    for article in soup.find_all('article'):
        article.find('revision').decompose()  # remove revision tag
        article_id: str = article.find('id').string  # get article id
        article_title: str = article.find('title').string  # get article id

        body_tokens = tokenize_body(article)
        category_tokens = tokenize_categories(article)

        article_tokens: [str] = [article_id, *body_tokens, *category_tokens]
        # print('\nArticle: {}\nTokens: {}\n'.format(article_id, article_tokens))
        token_occurrences = {}
        list(map(lambda token: insert_index(article_id, token, token_occurrences), article_tokens))
        article_table[article_id] = [article_title, file.name, soup.index(article), token_occurrences, len(body_tokens)]
        avg_dl += len(body_tokens)
    avg_dl /= len(article_table)
    pass


def tokenize_categories(article) -> [str]:
    return list(flat_map(lambda category: tokenization(category.string), article.find_all('category')))


def tokenize_body(article) -> [str]:
    article_body = article.find('bdy')
    if article_body is None:
        return []
    return tokenization(article_body.string)


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


def query(query_string: str, eval_type):
    tokens = text2tokens(query_string)
    results = {}
    interesting_article_ids = set()
    for token in tokens:
        interesting_article_ids.update(inverted_index_table[token])

        for key in interesting_article_ids:
            if eval_type == 'bm25':
                results[key] = bm25(tokens, key, article_table[key])
            elif eval_type == 'tf-idf':
                results[key] = tf_idf(tokens, key, article_table[key])
            else:
                print('invalid evalaution type, enter "tf-idf" or "bm25"')
                return

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        count = 0
        for key, value in sorted_results:
            count += 1
            print("#{} is article {} with score {} and title {}".format(count, key, value, article_table[key][0]))
            if count >= 100:
                return


def bm25(query_tokens: [str], article_id, article_stats):
    score = 0
    for qi in query_tokens:
        if qi not in article_stats[3]:
            continue
        idf_val = idf(qi)
        f = article_stats[3][qi]  # qi s frequency in D
        k = (1.2 + 2) / 2
        b = 0.75
        dl = article_stats[4]
        nominator = f * (k + 1)
        denominator = f + k * (1 - b + (b * (dl / avg_dl)))
        score += idf_val * (nominator / denominator)
    return score


def idf(qi):
    N = len(article_table)
    n_qi = len(inverted_index_table[qi])
    nominator = N - n_qi + 0.5
    denominator = n_qi + 0.5
    return np.log((nominator / denominator) + 1)


def tf_idf(query_tokens: [str], article_id, article_stats):
    score = 0
    for qi in query_tokens:
        if qi not in article_stats[3]:
            continue
        idf_val = idf(qi)
        f = article_stats[3][qi]
        score += idf_val * f
    return score


if __name__ == '__main__':
    forceReIndex = False
    if forceReIndex or not load_tables():
        start_time = time.time()
        load_wiki_files()
        print("re-indexing took --- %s seconds ---" % (time.time() - start_time))
        save_tables()
    else:
        print("indexes loaded from memory")

    query("b", 'tf-idf')

    query_start_time = time.time_ns()
    query("Freestyle", 'tf-idf')
    query_end_time = time.time_ns()
    print("--- Query took %s milliseconds ---" % ((query_end_time - query_start_time) / 1000000.0))

import numpy as np
import time

from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared import get_english_model, process_text, remove_stop_words
from typing import List, Tuple

_dataset_file_path = '../dataset.tsv'


def load_dataset(include_stop_words: bool) -> List[Tuple[float, List[str], List[str]]]:
    print('Preparing dataset')
    start_time = time.time()
    dataset = []
    with open(_dataset_file_path) as file:
        lines = file.readlines()
        for line in lines:
            [ground_truth, text1, text2] = line.strip('\n').split('\t')
            part = (
                float(ground_truth), prepare_text(text1, include_stop_words), prepare_text(text2, include_stop_words)
            )
            dataset.append(part)
    print(f'Dataset prepared in {round(time.time() - start_time, 2)} seconds\n')
    return dataset


def prepare_text(text: str, include_stop_words: bool) -> List[str]:
    tokens = process_text(text)
    if not include_stop_words:
        tokens = remove_stop_words(tokens)
    model = get_english_model()
    return list(filter(lambda word: model.has_index_for(word), tokens))


def vector_space_predictions(dataset: List[Tuple[float, List[str], List[str]]]) -> List[float]:
    results = []
    for (ground_truth, text1, text2) in dataset:
        vectors = TfidfVectorizer().fit_transform([' '.join(text1), ' '.join(text2)])
        prediction = cosine_similarity(vectors[0], vectors[1])[0][0]
        results.append(prediction)
    return results


def mean_short_text_vector_predictions(dataset: List[Tuple[float, List[str], List[str]]]) -> List[float]:
    results = []
    for (ground_truth, text1, text2) in dataset:
        first_vector = mean_average_vector(text1)
        second_vector = mean_average_vector(text2)
        prediction = cosine_similarity([first_vector], [second_vector])[0][0]
        results.append(prediction)
    return results


def mean_average_vector(text: List[str]):
    model = get_english_model()
    vectors = []
    for word in text:
        vectors.append(model.get_vector(word, norm=True))
    result = np.mean(vectors, axis=0)
    return result


def idf_short_text_vector_predictions(dataset: List[Tuple[float, List[str], List[str]]]) -> List[float]:
    results = []
    for (ground_truth, text1, text2) in dataset:
        vectorizer = TfidfVectorizer(use_idf=True)
        vectorizer.fit_transform([' '.join(text1), ' '.join(text2)])
        idf_values = vectorizer.idf_
        mapping = vectorizer.vocabulary_
        first_vector = idf_weighted_average_vector(text1, idf_values, mapping)
        second_vector = idf_weighted_average_vector(text2, idf_values, mapping)
        prediction = cosine_similarity([first_vector], [second_vector])[0][0]
        results.append(prediction)
    return results


def idf_weighted_average_vector(text: List[str], idf_values, mapping):
    model = get_english_model()
    vectors = []
    for word in text:
        idf = idf_values[mapping[word]]
        vectors.append(np.multiply(model.get_vector(word, norm=True), idf))
    result = np.average(vectors, axis=0)
    return result


def evaluate(name: str, dataset: List[Tuple[float, List[str], List[str]]], predictions: List[float]):
    ground_truths = list(map(lambda entry: entry[0], dataset))
    pearson_score = pearsonr(ground_truths, predictions)
    print(f'{name} -> Pearson correlation: {pearson_score}')


def evaluate_processing(include_stop_words: bool):
    dataset = load_dataset(include_stop_words)
    evaluate('Vector space', dataset, vector_space_predictions(dataset))
    evaluate('Short text vector with mean average', dataset, mean_short_text_vector_predictions(dataset))
    evaluate('Short text vector with weighted average using IDF', dataset, idf_short_text_vector_predictions(dataset))


def main():
    print('~~~~~ Lower-casing, tokenization ~~~~~\n')
    evaluate_processing(include_stop_words=True)
    print()

    print('~~~~~ Lower-casing, tokenization, stop word removal ~~~~~\n')
    evaluate_processing(include_stop_words=False)
    print()


if __name__ == '__main__':
    main()

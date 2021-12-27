import time

from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared import process_text, remove_stop_words
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
            float(ground_truth), prepare_text(text1, include_stop_words), prepare_text(text2, include_stop_words))
            dataset.append(part)
    print(f'Dataset prepared in {round(time.time() - start_time, 2)} seconds\n')
    return dataset


def prepare_text(text: str, include_stop_words: bool) -> List[str]:
    tokens = process_text(text)
    if not include_stop_words:
        tokens = remove_stop_words(tokens)
    return tokens


def vector_space_predictions(dataset: List[Tuple[float, List[str], List[str]]]) -> List[float]:
    results = []
    for index, (ground_truth, text1, text2) in enumerate(dataset):
        vectors = TfidfVectorizer().fit_transform([' '.join(text1), ' '.join(text2)])
        prediction = cosine_similarity(vectors[0], vectors[1])[0][0]
        results.append(prediction)
    return results


def mean_short_text_vector_predictions(dataset: List[Tuple[float, List[str], List[str]]]) -> List[float]:
    results = []
    return results


def idf_short_text_vector_predictions(dataset: List[Tuple[float, List[str], List[str]]]) -> List[float]:
    results = []
    return results


def evaluate(name: str, dataset: List[Tuple[float, List[str], List[str]]], predictions: List[float]):
    ground_truths = list(map(lambda entry: entry[0], dataset))
    pearson_score = pearsonr(ground_truths, predictions)
    print(f'{name} -> Pearson correlation: {pearson_score}')


def evaluate_processing(include_stop_words: bool):
    dataset = load_dataset(include_stop_words)
    evaluate('Vector space', dataset, vector_space_predictions(dataset))
    # TODO
    # evaluate('Short text vector with mean average', dataset, mean_short_text_vector_predictions(dataset))
    # TODO
    # evaluate('Short text vector with weighted average using IDF', dataset, idf_short_text_vector_predictions(dataset))


def main():
    print('~~~~~ Lower-casing, tokenization ~~~~~\n')
    evaluate_processing(include_stop_words=True)
    print()

    print('~~~~~ Lower-casing, tokenization, stop word removal ~~~~~\n')
    evaluate_processing(include_stop_words=False)
    print()


if __name__ == '__main__':
    main()

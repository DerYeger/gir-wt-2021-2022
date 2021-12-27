import time

from scipy.stats.stats import pearsonr
from shared import process_text
from typing import List, Tuple

_dataset_file_path = '../dataset.tsv'


def load_dataset() -> List[Tuple[float, str, str]]:
    print('Loading dataset')
    start_time = time.time()
    dataset = []
    with open(_dataset_file_path) as file:
        lines = file.readlines()
        for line in lines:
            [ground_truth, text1, text2] = line.strip('\n').split('\t')
            part = (float(ground_truth), process_text(text1), process_text(text2))
            dataset.append(part)
    print(f'Dataset loaded in {round(time.time() - start_time, 2)} seconds\n')
    return dataset


def vector_space_predictions(dataset: List[Tuple[float, str, str]]) -> List[float]:
    # TODO
    return []


def mean_short_text_vector_predictions(dataset: List[Tuple[float, str, str]]) -> List[float]:
    # TODO
    return []


def idf_short_text_vector_predictions(dataset: List[Tuple[float, str, str]]) -> List[float]:
    # TODO
    return []


def evaluate(dataset: List[Tuple[float, str, str]], predictions: List[float]):
    ground_truths = list(map(lambda entry: entry[0], dataset))
    pearson_score = pearsonr(ground_truths, predictions)
    print(f'Pearson correlation: {pearson_score}')


def main():
    dataset = load_dataset()
    evaluate(dataset, vector_space_predictions(dataset))
    evaluate(dataset, mean_short_text_vector_predictions(dataset))
    evaluate(dataset, idf_short_text_vector_predictions(dataset))


if __name__ == '__main__':
    main()

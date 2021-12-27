from shared import process_text
from typing import List, Tuple

_dataset_file_path = '../dataset.tsv'


def load_dataset() -> List[Tuple[float, str, str]]:
    dataset = []
    with open(_dataset_file_path) as file:
        lines = file.readlines()
        for line in lines:
            [ground_truth, text1, text2] = line.strip('\n').split('\t')
            part = (float(ground_truth), process_text(text1), process_text(text2))
            dataset.append(part)
    return dataset


def main():
    dataset = load_dataset()


if __name__ == '__main__':
    main()

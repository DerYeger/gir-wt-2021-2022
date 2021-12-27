from shared import model
from typing import Tuple


def get_cosine_similarity(pair: Tuple[str, str]):
    result = model.similarity(pair[0], pair[1])
    print(f'cos_sim({pair[0]}, {pair[1]}) = {result}\n')


def get_three_most_similar(word: str):
    print(f'Words similar to "{word}":')
    results = model.most_similar(word, topn=3)
    for i, (otherWord, score) in enumerate(results):
        print(f'{i + 1}. {otherWord} ({score})')
    print()


def main():
    for pair in [('cat', 'dog'), ('cat', 'Vienna'), ('Vienna', 'Austria'), ('Austria', 'dog')]:
        get_cosine_similarity(pair)

    for word in ['Vienna', 'Austria', 'cat']:
        get_three_most_similar(word)


if __name__ == '__main__':
    main()

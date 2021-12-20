import gensim
import time

model_path = '../model/wiki-news-300d-1M-subword.vec'
print('Loading model')
start_time = time.time()
model = gensim.models.keyedvectors.load_word2vec_format(model_path)
print(f'Model loaded in {round(time.time() - start_time, 2)} seconds\n')


def get_three_most_similar(word: str):
    print(f'Words similar to "{word}":')
    results = model.most_similar(word, topn=3)
    for i, (otherWord, score) in enumerate(results):
        print(f'{i + 1}. {otherWord} ({score})')
    print()


def main():
    get_three_most_similar('Vienna')
    get_three_most_similar('Austria')
    get_three_most_similar('cat')


if __name__ == '__main__':
    main()

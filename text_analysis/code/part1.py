from shared import get_cosine_similarity, get_english_model, get_three_most_similar


def main():
    model = get_english_model()
    for pair in [('cat', 'dog'), ('cat', 'Vienna'), ('Vienna', 'Austria'), ('Austria', 'dog')]:
        get_cosine_similarity(pair, model)

    for word in ['Vienna', 'Austria', 'cat']:
        get_three_most_similar(word, model)


if __name__ == '__main__':
    main()

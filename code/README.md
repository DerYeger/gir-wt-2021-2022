# GIR Search Engine

## Dataset

To use this search engine, download the [dataset](https://owncloud.tuwien.ac.at/index.php/s/LovQLx2A4FOxpwf) and place it in the `./dataset` directory.

You should have the following directory structure:

```
code
│   README.md
│   *.py 
│
└───dataset
│   │   aricle.dtd
│   │   eval.qrels
│   │   topics.xml
│   │
│   └───wikipedia articles
│       │   1.xml
│       │   2.xml
│       │   ...
│
└───retrieval_results
│   │   bm25-evaluation.txt
│   │   bm25-evaluation.txt.eval.txt
│   │   tf-idf-evaluation.txt
│   └   tf-idf-evaluation.txt.eval.txt
│   
└───tables
    │   article_table.txt
    │   average_word_count.txt
    └   inverted_index.npy
```

## Installation

To run the program, you need Python 3.9+ and the following Python libraries:

- [NumPy](https://numpy.org) (used for memory efficient storage)
- [PyInquirer](https://github.com/CITGuru/PyInquirer) (used for interactive CLI)
- [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) (used for CLI input validation)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) (used for document parsing)
- [NLTK](https://www.nltk.org) (used for stemming and providing a stop word list)
- [termcolor](https://pypi.org/project/termcolor/) (used for highlighting output)

## Usage

Run `python main.py` to start the program.
If no index is found in the `./tables` directory, a fresh index will be created.
Otherwise, it will load the index from the respective files.
Alternatively, you can run `python main.py -c` or `python main.py --clean` to always create a fresh index.
When a fresh index is being created, the program will ask how many files it is supposed to index.
If you want to index all files, enter `-1`.

### Evaluation mode

Select `Evaluation` in the menu to run the evaluation mode.
First, the program will parse the topics file `./dataset/topics.xml` and query the index for every topic twice using both *BM25* and *TF-IDF* scoring methods.
Afterward, it will attempt to run `trec_eval` for the evaluation results and save the scores in separate files.

### Exploration mode

Select `Exploration` to explore the index.
Chose a scoring method and enter your query.
The program will present a list of results in descending order.
Select an entry to view its contents or chose `Return` to skip this part.

### Other

Select `Reset index` to create a fresh index without restarting the program.

Select `Exit` to terminate the program.
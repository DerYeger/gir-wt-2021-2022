from inverted_index import InvertedIndex
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import prompt

_index_dir: str = './tables'
_dataset_dir: str = './dataset/articles'


def get_index(load_from_disk) -> InvertedIndex:
    max_files = -1
    if not load_from_disk:
        indexing_answers = prompt(_indexing_questions)
        max_files = indexing_answers.get('max_files')
        print()
    return InvertedIndex(_index_dir, _dataset_dir, max_files, load_from_disk=load_from_disk)


class NumberValidator(Validator):
    def validate(self, document):
        try:
            value = int(document.text)
            return value > 0 or value == -1
        except ValueError:
            raise ValidationError(message='Please enter a valid number larger',
                                  cursor_position=len(document.text))


_indexing_questions = [
    {
        'type': 'input',
        'name': 'max_files',
        'message': 'Amount of files to index (-1 for all files)',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    }
]

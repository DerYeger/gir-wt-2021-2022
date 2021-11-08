from inverted_index import InvertedIndex
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import prompt

_index_dir: str = './tables'
_dataset_dir: str = './dataset/articles'


def get_index(load_from_disk) -> InvertedIndex:
    def get_max_file_count():
        indexing_answers = prompt(_indexing_questions)
        max_files = indexing_answers.get('max_files')
        print()
        return max_files

    return InvertedIndex(_index_dir, _dataset_dir, load_from_disk=load_from_disk, get_max_file_count=get_max_file_count)


class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(message='Please enter a valid number', cursor_position=len(document.text))


_indexing_questions = [
    {
        'type': 'input',
        'name': 'max_files',
        'message': 'Amount of files to index (-1 for all files)',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    }
]

if __name__ == '__main__':
    InvertedIndex(_index_dir, _dataset_dir, load_from_disk=False, get_max_file_count=lambda: 3)

from termcolor import colored


def error(text) -> str:
    return colored(text, 'red')


def highlight(text) -> str:
    return colored(str(text), 'cyan')


def info(text: str) -> str:
    return colored(text, 'green')


def path_color(text: str) -> str:
    return colored(text, 'blue')

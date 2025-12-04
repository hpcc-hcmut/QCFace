import os

def convert_backslashes(string: str):
    return string.replace("\\", "/")


def path_join(*paths):
    return convert_backslashes(os.path.join(*paths))

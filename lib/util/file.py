import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path




def remove_dir(path):
    for file in os.listdir(path):
        file = f'{path}/{file}'
        if os.path.isfile(file):
            os.remove(file)
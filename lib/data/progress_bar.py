from tqdm.notebook import tqdm_notebook

def progress_bar(count, title='Processing'): 
    return tqdm_notebook(total=count, desc = title)
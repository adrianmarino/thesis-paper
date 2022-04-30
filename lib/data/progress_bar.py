from alive_progress import alive_bar


def progress_bar(count, title=''):
    return alive_bar(
        count, 
        title=title, 
        force_tty=True, 
        bar='blocks', 
        spinner='twirls'
    )

def update_bar(bar, index, times=5000):
    if index % times == 0:
        bar(times)
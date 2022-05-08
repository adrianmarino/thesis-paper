import seaborn as sns
import matplotlib.pyplot as plt
from .diagrams import barplot, histplot, words_clous_plot


def describe_num_var(df, column, bins='auto', stat='count', title='', show_table=False, show_range=False):
    if show_range or show_table:
        df_column = df[[column]]

    if show_range:
        column_range = (min(df_column.values)[0], max(df_column.values)[0])
        print(f'\nRange: {column_range}\n')  

    histplot(df, column, bins, stat=stat, title=title)

    if show_table:
        print('\nMetrics:\n')
        display(df_column.describe())

def array_num_to_str(numbers): return [str(n) for n in numbers]


def describe_cat_var(df, column, max_shown_values=10, order_column=None, asc_order=False, show_table=False):   
    if order_column == None:
        order_column = 'count'
    
    count_by_value = df \
    .groupby(column) \
    .size() \
    .reset_index(name='count') \
    .sort_values(by=order_column, ascending=asc_order) \
    .reset_index(drop=True)

    if show_table:
        print(f'\nFrecuency table:\n')
        display(count_by_value)

    barplot(count_by_value, x=column, y='count', title=f'{column} categorical variable')


def describe_text_var(df, column, flatten=False):
    values = df[column].values.tolist()
        
    if flatten:
        values = [vv for v in values for vv in v]

    words_clous_plot(values, title = f'{column} text variable')
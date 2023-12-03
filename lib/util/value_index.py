import pandas as pd


class ValueIndex:
    """Allows to filter a pandas DataFrame using multiple column values
        like a regular pandas filtering but using a list of tuples as filter values.

        i.e:
            index_tuples = [(1, 2), (3, 4)]

            df:
                |user_id | movie_id | rating |
                |      1 |        2 |      5 |
                |      3 |        4 |      1 |
                |      2 |        1 |      3 |

            valueIndex = MultiIndex(df, 'rating', [user_id, movie_id])

            filtered_df = valueIndex[index_tuples]

            filtered_df:
                |user_id | movie_id | rating |
                |      1 |        2 |      5 |
                |      3 |        4 |      1 |

    """
    def __init__(
        self,
        df,
        value_col,
        index_cols
    ):
        index = pd.MultiIndex.from_arrays(
            [df[col].values for col in index_cols],
            names=index_cols
        )
        self.df = pd.DataFrame({value_col: df[value_col].values}, index=index)
        self.value_col = value_col

    def __getitem__(self, index_tuples):
        return self.df.loc[index_tuples][self.value_col].values

    def __repr__(self): return str(self.df)
    def __str__(self): return  str(self.df)
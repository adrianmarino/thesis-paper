from sklearn.preprocessing import StandardScaler
import pandas as pd


class Normalizer:
    def __init__(self, df, columns):
        self.scaler = StandardScaler()
        self.scaler.fit(df[columns])
        self.norm_columns = columns
        self.non_norm_columns = list(set(df.columns) - set(columns))

    def __call__(self, input_df):
        input_norm_df = pd.DataFrame(
            self.scaler.transform(input_df[self.norm_columns]),
            columns=self.norm_columns
        )

        for c in self.non_norm_columns:
            input_norm_df[c] = input_df[c].values

        return input_norm_df
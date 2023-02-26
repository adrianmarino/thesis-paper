import util as ut


def n_iqr_range(df, column,  n = 1.5 ):
    stats = df[column].describe()

    Q3 = stats[stats.index == '75%'].values[0]
    Q1 = stats[stats.index == '25%'].values[0]
    IQR = Q3 - Q1

    lower_bound = ut.clamp(Q1 - (n * IQR))
    upper_bound = Q3 + (n * IQR)

    return lower_bound, upper_bound

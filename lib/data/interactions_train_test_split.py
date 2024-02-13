
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


def interactions_train_test_split(
    dataset,
    order_col          = 'timestamp',
    user_id_col        = 'user_id',
    item_id_col        = 'item_id',
    n_min_interactions = 20,
    test_size          = 0.3
):
    """
        For each user, take interactions ascending ordered by 'order_col' and 
        split these into user train-test sets. Next reduce user sets into main
        train-test groups. Finally, remove test set items that non exist into
        train set. Take in account users with more than 'n_min_interactions' and
        split user interactions using 'test_size' percent.
    """
    groups = dataset \
        .sort_values(order_col) \
        .groupby(user_id_col)

    train_set, test_set = [], []

    for user_id in dataset[user_id_col].unique():
        user_interactions = groups.get_group(user_id)

        if user_interactions.shape[0] >= n_min_interactions:
            user_train, user_test = train_test_split(
                user_interactions, 
                test_size=test_size,
            )
        
            train_set.append(user_train)
            test_set.append(user_test)
            
    train_df = pd.concat(train_set)
    test_df  = pd.concat(test_set)


    """ 
    Include only items that exists in train set.
    """
    test_df = test_df[
        (test_df[user_id_col].isin(train_df[user_id_col].values)) &
        (test_df[item_id_col].isin(train_df[item_id_col].values))
    ]

    logging.info(f'Train: {(len(train_df)/len(dataset))*100:.2f} % - Test: {(len(test_df)/len(dataset))*100:.2f} %')

    return train_df, test_df
from math import log2


def idcg(ratings):
  return dcg(sorted(ratings, reverse=True))


def dcg(ratings):
  return sum([rating / log2(position+1) for position, rating in enumerate(ratings, start=1)])


def ndcg(ratings):
  return dcg(ratings)/idcg(ratings)



def recall(
    recommended_items,
    relevant_items
):
    # Calculate the intersection of recommended_items and relevant_itemsPrecision
    true_positive = len(set(recommended_items).intersection(set(relevant_items)))

    # Calculate the total number of relevant items
    total_relevant_items = len(relevant_items)

    return true_positive / total_relevant_items if total_relevant_items > 0 else 0


def mean_reciprocal_rank(
    recommended_items_list,
    relevant_items_list
):
    if len(recommended_items_list) != len(relevant_items_list):
        raise ValueError("The length of recommended_items_list and relevant_items_list must be the same.")

    reciprocal_ranks = []

    # Iterate through the lists of recommended items and relevant items for each user
    for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list):
        # Find the reciprocal rank for each user
        for rank, item in enumerate(recommended_items, start=1):
            if item in relevant_items:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def average_precision(
    recommended_items,
    relevant_items
):
    true_positives = 0
    sum_precisions = 0

    for rank, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            true_positives += 1
            precision_at_rank = true_positives / rank
            sum_precisions += precision_at_rank

    return sum_precisions / len(relevant_items) if len(relevant_items) > 0 else 0


def mean_average_precision(
    recommended_items_list,
    relevant_items_list
):
    if len(recommended_items_list) != len(relevant_items_list):
        raise ValueError("The length of recommended_items_list and relevant_items_list must be the same.")

    average_precisions = []

    # Calculate the average precision for each user
    for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list):
        ap = average_precision(recommended_items, relevant_items)
        average_precisions.append(ap)

    # Calculate the mean average precision across all users
    map_value = sum(average_precisions) / len(average_precisions)
    return round(map_value, 2)


# By user and total.
def catalog_coverage(
    recommended_items_list,
    catalog_items
):
    # Flatten the list of recommended items and convert it to a set
    unique_recommended_items = set(item for sublist in recommended_items_list for item in sublist)

    # Calculate the intersection of unique recommended items and catalog items
    covered_items = unique_recommended_items.intersection(catalog_items)

    return len(covered_items) / len(catalog_items)


def serendipity(
    recommended_items_list,
    relevant_items_list,
    popular_items,
    k = 10
):
    serendipity_score = 0
    total_users = len(recommended_items_list)

    for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list):
        # Select the top-k recommended items
        top_k_recommended = recommended_items[:k]

        # Find the serendipitous items by removing popular items from relevant items
        serendipitous_items = set(relevant_items) - set(popular_items)

        # Count the number of serendipitous items in the top-k recommendations
        serendipitous_recommendations = len(set(top_k_recommended) & serendipitous_items)

        # Calculate the proportion of serendipitous items in the top-k recommendations
        serendipity_score += serendipitous_recommendations / k

    return serendipity_score / total_users
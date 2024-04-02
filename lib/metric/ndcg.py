from math import log2


def idcg(ratings):
  return dcg(sorted(ratings, reverse=True))


def dcg(ratings):
  return sum([rating / log2(position+1) for position, rating in enumerate(ratings, start=1)])


def ndcg(ratings):
  return dcg(ratings)/idcg(ratings)

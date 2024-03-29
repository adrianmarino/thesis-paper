import math


def idcg(ratings):
  return dcg(sorted(ratings, reverse=True))


def dcg(ratings):
  return sum([rating / math.log2(idx+2) for idx, rating in enumerate(ratings)])


def ndcg(ratings):
  return dcg(ratings)/idcg(ratings)

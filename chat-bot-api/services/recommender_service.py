from models import RecommenderTrainResult
import util as ut


class RecommenderService:
  def __init__(self, ctx):
    self.ctx = ctx

  def train(self):
    return RecommenderTrainResult()
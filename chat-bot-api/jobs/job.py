import service as srv
import os
import util as ut
from abc import ABC, abstractmethod


class Job(ABC):

  def __init__(
    self,
    ctx,
    name,
    trigger_fn = None
  ):
    self.ctx = ctx
    self.trigger_fn     = trigger_fn
    self.tmp_path       = os.environ['TMP_PATH']
    self.cfg_file_path  = f'{self.tmp_path}/{name}'
    self.__cfg          = None


  @property
  def _cfg(self):
    if self.__cfg is None:
      self.__cfg = ut.Picket.try_load(self.cfg_file_path)
    return self.__cfg


  @_cfg.setter
  def _cfg(self, cfg):
    ut.Picket.save(self.cfg_file_path, cfg)
    self.__cfg = None


  @abstractmethod
  async def __call__(self):
    pass
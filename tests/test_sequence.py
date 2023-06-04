import sys
sys.path.append('./lib')

import warnings
warnings.filterwarnings('ignore')

import pytest
from data import Sequencer, check_sequence
import pandas as pd
import numpy as np


class TestSequence:
    def test_consecurive_sequence(self):
        # Prepare...
        df  = pd.DataFrame({ 'id': list(range(1, 1001, 5)) })
        sec = Sequencer('id', 'seq')

        # Perform
        df = sec.perform(df)

        # Assert
        sequence = np.unique(df['seq'].values)
        check_sequence(sequence)
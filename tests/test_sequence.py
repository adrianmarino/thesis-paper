import sys
sys.path.append('./lib')

import pytest
import data as dt
import pandas as pd
import numpy as np


class TestSequence:
    def test_consecurive_sequence(self):
        # Prepare...
        df  = pd.DataFrame({ 'id': list(range(1, 1001, 5)) })
        sec = dt.Sequencer('id', 'seq')

        # Perform
        df = sec.perform(df)

        # Assert
        sequence = np.unique(df['seq'].values)
        dt.check_sequence(sequence)
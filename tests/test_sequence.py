import sys
sys.path.append('./lib')

import pytest
import data as dt
import pandas as pd
import numpy as np


def check_sequence(values):
    last = -1
    for seq_value in values:
        if last == -1:
            last = seq_value
            continue

        assert last == seq_value-1
        last = seq_value


class TestSequence:
    def test_consecurive_sequence(self):
        # Prepare...
        df  = pd.DataFrame({ 'id': list(range(1, 1001, 5)) })
        seq = dt.Sequencer()

        # Perform
        df['seq'] = df.id.apply(seq.get)

        # Assert
        sequence = np.unique(df['seq'].values)
        check_sequence(sequence)
import logging
import numpy as np


def sequences_to_feats_target(sequences, feat_seq_len, mask = 0):
    """
    Breaks item sequences to pairs <prev_items_seq, next_item>
    """
    features, targets = [], []
    for sequence in sequences:                
        if len(sequence) > feat_seq_len -1:
            try:
                features.append(np.concatenate([sequence[:feat_seq_len-1], np.array([mask])]))
                targets.append(sequence[feat_seq_len-1])
            except Exception as e:
                logging.error(f'Error when separate esquence into features, target. sequence: {type(sequence)} {sequence}', e)

            _features, _target = sequences_to_feats_target([sequence[1:]], feat_seq_len, mask)
            features.extend(_features)
            targets.extend(_target)

    return np.array(features), np.array(targets)
import sys
sys.path.append('./lib')

import numpy as np
import pytest
import metric.discretizer as dr


def test_mapping():
    # Prepare
    map = { 
        0.5: 1,
        1  : 2,
        1.5: 3,
        2  : 4,
        2.5: 5,
        3  : 6,
        3.5: 7,
        4  : 8,
        4.5: 9,
        5  : 10,
    }
    discretizer = dr.mapping(map).closure()

    for key, value in map.items():
        assert discretizer(key) == value


def test_sequence():
    # Prepare
    values = np.array([0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0])
    sequence = list(range(len(values)))

    discretizer = dr.sequence(values).closure()

    for value, seq in zip(values, sequence):
        assert discretizer(value) == seq


def test_rounder():
    # Prepare
    discretizer = dr.rounder(0.5, max_value=5.0).closure()


    # X <= 4.25 --> 4
    assert discretizer(4)    == 4
    assert discretizer(4.25) == 4

    # 4.25 < X <= 4.75 --> 4.5 (+/- 0.25)
    assert discretizer(4.26) == 4.5
    assert discretizer(4.3)  == 4.5
    assert discretizer(4.74) == 4.5

    # X > 4.75 --> 5
    assert discretizer(4.75) == 5

    assert discretizer(0.0) == 0.5
    assert discretizer(0.5) == 0.5
    assert discretizer(1.0) == 1.0
    assert discretizer(1.5) == 1.5
    assert discretizer(2.0) == 2.0
    assert discretizer(2.5) == 2.5
    assert discretizer(3.0) == 3.0
    assert discretizer(3.5) == 3.5
    assert discretizer(4.0) == 4.0
    assert discretizer(4.5) == 4.5
    assert discretizer(5.0) == 5.0
    assert discretizer(5.5) == 5.0


def test_round_sequence():
    # Prepare
    values = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0])
    sequence = list(range(len(values)))

    discretizer = dr.round_sequence(values, 0.5).closure()

    for value, seq in zip(values, sequence):
        assert discretizer(value) == seq



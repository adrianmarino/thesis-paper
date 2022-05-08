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


def test_round_sequence():
    # Prepare
    values = np.array([0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])

    # Perform
    discretizer = dr.round_sequence(values).closure()

    # Asserts
    assert discretizer(0.0) == 0
    assert discretizer(0.5) == 0
    assert discretizer(1.0) == 1
    assert discretizer(1.5) == 3
    assert discretizer(2.0) == 3
    assert discretizer(2.5) == 3
    assert discretizer(3.0) == 5
    assert discretizer(3.5) == 7
    assert discretizer(4.0) == 7
    assert discretizer(4.5) == 7
    assert discretizer(5.0) == 9
    assert discretizer(5.5) == 9

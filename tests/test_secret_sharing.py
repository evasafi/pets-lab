import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secret_sharing import share, P


def test_shares_sum_to_value():
    for value in [0, 1, 37, 979, 10**9]:
        shares = share(value, 3)
        assert sum(shares) % P == value


def test_number_of_shares():
    assert len(share(42, 3)) == 3
    assert len(share(42, 5)) == 5


def test_shares_are_in_field():
    assert all(0 <= s < P for s in share(42, 3))


def test_same_value_gives_different_shares():
    assert share(42, 3) != share(42, 3)

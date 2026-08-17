import pytest

from grounded.agent.tools import calculate


def test_growth_percentage():
    assert abs(calculate("(48200-42300)/42300*100") - 13.9479) < 0.01


def test_basic_arithmetic():
    assert calculate("2 + 3 * 4") == 14.0


def test_rejects_code():
    with pytest.raises(Exception):
        calculate("__import__('os').system('echo hi')")

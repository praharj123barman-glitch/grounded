from eval.metrics import hit_at_k, mean, mrr, precision_at_k, recall_at_k


def test_recall_full_and_zero():
    texts = ["revenue was 48,200 million", "other", "cash 7,900"]
    assert recall_at_k(texts, ["48,200"], 5) == 1.0
    assert recall_at_k(texts, ["99,999"], 5) == 0.0


def test_recall_partial():
    texts = ["mentions apple", "mentions banana"]
    assert recall_at_k(texts, ["apple", "cherry"], 5) == 0.5


def test_mrr_positions():
    assert mrr(["nothing here", "apple found"], ["apple"]) == 0.5
    assert mrr(["apple found", "nothing"], ["apple"]) == 1.0


def test_precision():
    assert precision_at_k(["apple", "orange", "grape"], ["apple"], 3) == 1 / 3


def test_hit():
    assert hit_at_k(["x", "apple here"], ["apple"], 5) == 1.0
    assert hit_at_k(["x"], ["apple"], 5) == 0.0


def test_unanswerable_returns_none():
    assert recall_at_k(["x"], [], 5) is None
    assert mrr(["x"], []) is None


def test_mean_skips_none():
    assert mean([1.0, None, 0.0]) == 0.5

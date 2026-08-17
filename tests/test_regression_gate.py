from eval.regression_gate import compare


def test_gate_passes_when_stable():
    ok, failures = compare(
        {"faithfulness": 0.9, "recall_at_5": 0.8},
        {"faithfulness": 0.9, "recall_at_5": 0.8},
    )
    assert ok is True
    assert failures == []


def test_gate_fails_on_drop():
    ok, failures = compare({"faithfulness": 0.5}, {"faithfulness": 0.9})
    assert ok is False
    assert failures

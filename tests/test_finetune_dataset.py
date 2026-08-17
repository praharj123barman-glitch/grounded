from finetune.build_dataset import build


def test_build_dataset_from_golden():
    rows = build()
    assert len(rows) >= 10
    assert all("instruction" in r and "response" in r for r in rows)
    # unanswerable golden items are excluded from the training pairs
    assert all(r["response"] != "Not stated in the document." for r in rows)

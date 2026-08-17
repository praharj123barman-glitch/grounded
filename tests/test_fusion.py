from grounded.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_merges_all_ids():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ids = [x for x, _ in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    assert ids[0] in {"a", "b"}  # the two that rank high in both lists


def test_rrf_scores_descending():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)

from statbank_rag_agent.retrieval import reciprocal_rank_fusion


def test_table_ranked_well_by_both_methods_wins():
    vector = ["A", "B", "C"]
    keyword = ["B", "D", "A"]
    fused = reciprocal_rank_fusion([vector, keyword], k=4)
    assert fused[0] in {"A", "B"}  # each appears in both lists
    assert set(fused) == {"A", "B", "C", "D"}


def test_table_found_by_only_one_method_is_kept():
    fused = reciprocal_rank_fusion([["A", "B"], ["C"]], k=3)
    assert "C" in fused

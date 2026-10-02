from statbank_rag_agent.retrieval import (
    _year_span,
    filter_by_years,
    reciprocal_rank_fusion,
    years_in,
)


def test_table_ranked_well_by_both_methods_wins():
    vector = ["A", "B", "C"]
    keyword = ["B", "D", "A"]
    fused = reciprocal_rank_fusion([vector, keyword], k=4)
    assert fused[0] in {"A", "B"}  # each appears in both lists
    assert set(fused) == {"A", "B", "C", "D"}


def test_table_found_by_only_one_method_is_kept():
    fused = reciprocal_rank_fusion([["A", "B"], ["C"]], k=3)
    assert "C" in fused


def test_years_in_question():
    assert years_in("What was the population of Denmark in 1950?") == [1950]
    assert years_in("How many people live in Aarhus?") == []


def test_period_formats_give_the_year():
    assert _year_span("1901", "2026") == (1901, 2026)
    assert _year_span("2008Q1", "2026Q3") == (2008, 2026)
    assert _year_span("2021M10", "2026M08") == (2021, 2026)


def test_filter_keeps_only_tables_covering_the_year():
    spans = {"BEFOLK2": (1901, 2026), "BEFOLK3": (2008, 2026), "FOLK1A": (2008, 2026)}
    assert filter_by_years(["BEFOLK3", "FOLK1A", "BEFOLK2"], [1950], spans) == [
        "BEFOLK2"
    ]


def test_filter_falls_back_when_nothing_covers_the_year():
    spans = {"A": (2008, 2026)}
    assert filter_by_years(["A"], [1850], spans) == ["A"]


def test_covers_requires_every_year():
    from statbank_rag_agent.retrieval import covers

    assert covers((1901, 2026), [1950])
    assert not covers((2008, 2026), [1950])
    assert not covers((1990, 2000), [1995, 2010])
    assert not covers(None, [1950])

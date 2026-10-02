from statbank_rag_agent.catalogue import build_search_text, to_row

BEFOLK2 = {
    "id": "BEFOLK2",
    "text": "Population 1. January",
    "unit": "Number",
    "updated": "2026-02-11T08:00:00",
    "firstPeriod": "1901",
    "latestPeriod": "2026",
    "variables": ["sex", "age", "time"],
}
BEFOLK3 = {
    **BEFOLK2,
    "id": "BEFOLK3",
    "firstPeriod": "2008",
    "variables": ["region", "sex", "age", "time"],
}


def test_tables_with_the_same_title_get_different_search_texts():
    assert BEFOLK2["text"] == BEFOLK3["text"]
    assert build_search_text(BEFOLK2) != build_search_text(BEFOLK3)


def test_search_text_contains_variables_and_period():
    text = build_search_text(BEFOLK3)
    assert "region" in text
    assert "2008 to 2026" in text


def test_update_time_is_read_as_copenhagen_time():
    updated = to_row(BEFOLK2)["source_updated"]
    assert updated.utcoffset().total_seconds() == 3600  # February: UTC+1

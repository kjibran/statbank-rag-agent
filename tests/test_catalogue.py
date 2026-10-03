from statbank_rag_agent.catalogue import (
    MAX_VALUES_PER_VARIABLE,
    build_rich_search_text,
    build_search_text,
    to_row,
)

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


def tableinfo(industry_values: list[str]) -> dict:
    """Minimal tableinfo for a business survey, differing only in industry values."""
    return {
        "text": "Assessments about the business situation",
        "unit": "Per cent",
        "variables": [
            {
                "text": "industry",
                "time": False,
                "values": [
                    {"id": str(i), "text": v} for i, v in enumerate(industry_values)
                ],
            },
            {
                "text": "time",
                "time": True,
                "values": [{"id": "2026M09", "text": "2026M09"}],
            },
        ],
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


def test_rich_text_separates_tables_that_differ_only_in_values():
    retail = build_rich_search_text(
        tableinfo(["Retail trade", "Food stores"]), "2021M05", "2026M09"
    )
    building = build_rich_search_text(
        tableinfo(["Construction", "Civil engineering"]), "2021M05", "2026M09"
    )
    assert retail != building
    assert "Retail trade" in retail


def test_rich_text_skips_time_values_and_caps_long_lists():
    labels = [f"Value {i}" for i in range(MAX_VALUES_PER_VARIABLE + 5)]
    text = build_rich_search_text(tableinfo(labels), "2021M05", "2026M09")
    assert "2026M09." not in text.split("Unit")[1]  # time values are not listed
    assert "and 5 more" in text
    assert f"Value {MAX_VALUES_PER_VARIABLE}" not in text


def test_time_resolution_from_period_format():
    from statbank_rag_agent.catalogue import time_resolution

    assert "Monthly" in time_resolution("2026M08")
    assert "Quarterly" in time_resolution("2026Q3")
    assert "Quarterly" in time_resolution("2026K3")
    assert "Yearly" in time_resolution("2026")
    assert time_resolution("2019/2020") is None


def test_rich_text_says_monthly_for_monthly_tables():
    info = {
        "text": "Population at the first day of the month",
        "unit": "Number",
        "variables": [],
    }
    assert "Monthly" in build_rich_search_text(info, "2021M10", "2026M08")

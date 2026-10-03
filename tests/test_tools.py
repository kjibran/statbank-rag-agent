from statbank_rag_agent.tools import check_selections, is_total, match_values, run_tool

REGIONS = [
    {"id": "000", "text": "All Denmark"},
    {"id": "101", "text": "Copenhagen"},
    {"id": "751", "text": "Aarhus"},
    {"id": "461", "text": "Odense"},
]
INFO = {
    "id": "FOLK1A",
    "variables": [
        {"id": "OMRÅDE", "text": "region", "elimination": True, "values": REGIONS},
        {
            "id": "Tid",
            "text": "time",
            "elimination": False,
            "time": True,
            "values": [
                {"id": f"20{y}K{q}", "text": f"20{y}Q{q}"}
                for y in range(10, 27)
                for q in range(1, 5)
            ],
        },
    ],
}


def test_danish_spelling_variants_match():
    assert match_values(REGIONS, "Århus")[0]["code"] == "751"
    assert match_values(REGIONS, "aarhus")[0]["code"] == "751"


def test_totals_are_flagged():
    assert is_total("000", "All Denmark")
    assert is_total("TOT", "Total")
    assert not is_total("751", "Aarhus")
    assert match_values(REGIONS, "All Denmark")[0].get("is_total") is True


def test_valid_selection_passes():
    assert check_selections(INFO, {"OMRÅDE": ["751"], "Tid": ["2026K3"]}) is None


def test_time_must_be_selected():
    assert "must be selected" in check_selections(INFO, {"OMRÅDE": ["751"]})


def test_unknown_variable_and_code_are_explained():
    assert "Unknown variable" in check_selections(
        INFO, {"REGION": ["751"], "Tid": ["2026K3"]}
    )
    assert "Unknown codes" in check_selections(
        INFO, {"OMRÅDE": ["999"], "Tid": ["2026K3"]}
    )


def test_oversized_request_is_refused():
    all_quarters = [v["id"] for v in INFO["variables"][1]["values"]]
    problem = check_selections(
        INFO, {"OMRÅDE": ["000", "101", "751", "461"], "Tid": all_quarters}
    )
    assert "limit" in problem


def test_bad_tool_calls_return_errors_not_crashes():
    assert "error" in run_tool("no_such_tool", "{}")
    assert "error" in run_tool("describe_table", "not json")

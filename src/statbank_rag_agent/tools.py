import difflib
import json
import math
import unicodedata

from statbank_rag_agent.db import connect
from statbank_rag_agent.retrieval import hybrid_search, with_year_prefilter
from statbank_rag_agent.statbank import fetch_data

SEARCH_RESULTS = 8
EXAMPLE_VALUES = 5
MAX_MATCHES = 10
MAX_CELLS = 200
TOTAL_CODES = {"TOT", "IALT", "000"}

_search = with_year_prefilter(hybrid_search)


# --- Helpers (pure functions, tested without a database) ---


def is_total(code: str, label: str) -> bool:
    """Values like 'All Denmark' or 'Total' that sum other values in the same variable."""
    low = label.strip().lower()
    return (
        code.upper() in TOTAL_CODES
        or low in {"total", "all denmark", "in total"}
        or low.startswith("total")
        or low.endswith(" total")  # also covers labels ending in ", total"
    )


def _normalize(text: str) -> str:
    """Lowercase, Danish letters spelled out, accents removed. 'Århus' and 'Aarhus' match."""
    text = text.lower().replace("å", "aa").replace("æ", "ae").replace("ø", "oe")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def match_values(values: list[dict], text: str, limit: int = MAX_MATCHES) -> list[dict]:
    """Values whose label matches the text: exact first, then contained, then similar spelling."""
    query = _normalize(text)
    scored = []
    for value in values:
        label = _normalize(value["text"])
        if label == query:
            score = 3.0
        elif query in label:
            score = 2.0
        else:
            ratio = difflib.SequenceMatcher(None, query, label).ratio()
            score = ratio if ratio >= 0.75 else 0.0
        if score:
            scored.append((score, value))
    scored.sort(key=lambda pair: -pair[0])
    return [_value_entry(value) for _, value in scored[:limit]]


def _value_entry(value: dict) -> dict:
    entry = {"code": value["id"], "label": value["text"]}
    if is_total(value["id"], value["text"]):
        entry["is_total"] = True
    return entry


def check_selections(info: dict, selections: dict[str, list[str]]) -> str | None:
    """Return a problem description, or None if the selection can be fetched."""
    by_id = {v["id"].lower(): v for v in info["variables"]}
    for variable_id, codes in selections.items():
        variable = by_id.get(variable_id.lower())
        if variable is None:
            valid = ", ".join(v["id"] for v in info["variables"])
            return f"Unknown variable '{variable_id}'. Valid variables: {valid}."
        if not codes:
            return f"No codes given for {variable['id']}."
        valid_codes = {v["id"] for v in variable["values"]}
        unknown = [c for c in codes if c not in valid_codes]
        if unknown:
            return f"Unknown codes for {variable['id']}: {unknown[:5]}. Use find_values to look them up."

    selected = {k.lower() for k in selections}
    for variable in info["variables"]:
        if (
            not variable.get("elimination", False)
            and variable["id"].lower() not in selected
        ):
            return f"Variable {variable['id']} ({variable['text']}) must be selected."

    cells = math.prod(len(codes) for codes in selections.values())
    if cells > MAX_CELLS:
        return f"This would return {cells} numbers (limit {MAX_CELLS}). Select fewer values."
    return None


# --- Data access ---


def _tableinfo(table_id: str) -> dict | None:
    with connect(read_only=True) as conn:
        row = conn.execute(
            "select tableinfo from statbank_tables where table_id = %s",
            (table_id.upper(),),
        ).fetchone()
    return row[0] if row else None


def _unknown_table(table_id: str) -> dict:
    return {
        "error": f"Unknown table '{table_id}'. Use search_tables to find table ids."
    }


# --- The tools ---


def search_tables(query: str) -> dict:
    """Candidate tables for a topic."""
    table_ids = _search(query, SEARCH_RESULTS)
    if not table_ids:
        return {"tables": [], "note": "No tables found. Try different words."}
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "select table_id, title, first_period, latest_period, variables "
            "from statbank_tables where table_id = any(%s)",
            (table_ids,),
        ).fetchall()
    by_id = {r[0]: r for r in rows}
    return {
        "tables": [
            {
                "table_id": t,
                "title": by_id[t][1],
                "period": f"{by_id[t][2]} to {by_id[t][3]}",
                "variables": by_id[t][4],
            }
            for t in table_ids
            if t in by_id
        ]
    }


def describe_table(table_id: str) -> dict:
    """A table's variables, with example values, and the rules for selecting them."""
    info = _tableinfo(table_id)
    if info is None:
        return _unknown_table(table_id)
    variables = []
    for variable in info["variables"]:
        values = variable.get("values", [])
        entry = {
            "id": variable["id"],
            "label": variable["text"],
            "number_of_values": len(values),
            "can_be_left_out": variable.get("elimination", False),
        }
        if variable.get("time"):
            entry["is_time"] = True
            entry["earliest_period"] = values[0]["id"] if values else None
            entry["latest_periods"] = [v["id"] for v in values[-EXAMPLE_VALUES:]]
        else:
            entry["examples"] = [_value_entry(v) for v in values[:EXAMPLE_VALUES]]
        variables.append(entry)
    return {
        "table_id": info.get("id", table_id.upper()),
        "title": info.get("text"),
        "unit": info.get("unit"),
        "variables": variables,
        "rules": (
            "Variables that can be left out are summed over when not selected. "
            "The time variable must always be selected. "
            "Never add a total value to its own parts."
        ),
    }


def find_values(table_id: str, variable_id: str, text: str) -> dict:
    """Codes in one variable whose labels match the text, such as a municipality name."""
    info = _tableinfo(table_id)
    if info is None:
        return _unknown_table(table_id)
    variable = next(
        (v for v in info["variables"] if v["id"].lower() == variable_id.lower()), None
    )
    if variable is None:
        valid = ", ".join(v["id"] for v in info["variables"])
        return {
            "error": f"Table {table_id} has no variable '{variable_id}'. Valid variables: {valid}."
        }
    matches = match_values(variable.get("values", []), text)
    if not matches:
        return {
            "variable": variable["id"],
            "matches": [],
            "note": "No match. Try another spelling.",
        }
    return {"variable": variable["id"], "matches": matches}


def get_data(table_id: str, selections: dict[str, list[str]]) -> dict:
    """Fetch numbers for the selected codes, after checking the request."""
    info = _tableinfo(table_id)
    if info is None:
        return _unknown_table(table_id)
    problem = check_selections(info, selections)
    if problem:
        return {"error": problem}
    canonical = {v["id"].lower(): v["id"] for v in info["variables"]}
    rows = fetch_data(
        info["id"], {canonical[k.lower()]: codes for k, codes in selections.items()}
    )
    return {
        "table_id": info["id"],
        "unit": info.get("unit"),
        "rows": rows,
        "source": f"Statistics Denmark, StatBank.dk/{info['id'].lower()}",
    }


# --- Wiring for the LLM ---

TOOLS = {
    "search_tables": search_tables,
    "describe_table": describe_table,
    "find_values": find_values,
    "get_data": get_data,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": "Find Statistics Denmark tables about a topic. Returns candidates with title, period and variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The topic, for example 'population by municipality'. Leave out place names "
                            "and specific values, find those later with find_values. Include a year only "
                            "if the question is about a specific year."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Show a table's variables, example values, time periods and selection rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string", "description": "For example FOLK1A."}
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_values",
            "description": "Look up the codes for values in one variable, for example a municipality name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "variable_id": {
                        "type": "string",
                        "description": "Variable id from describe_table, for example OMRÅDE.",
                    },
                    "text": {
                        "type": "string",
                        "description": "What to look for, for example Aarhus.",
                    },
                },
                "required": ["table_id", "variable_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "Fetch numbers. Select codes per variable. Variables left out are summed over.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "selections": {
                        "type": "object",
                        "description": 'Variable id to list of codes, for example {"OMRÅDE": ["751"], "Tid": ["2026K3"]}.',
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "required": ["table_id", "selections"],
            },
        },
    },
]


def run_tool(name: str, arguments: str) -> dict:
    """Run a tool requested by the model. Problems come back as error messages, never crashes."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "The tool arguments were not valid JSON."}
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}."}
    try:
        return tool(**args)
    except TypeError as exc:
        return {"error": f"Wrong arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}

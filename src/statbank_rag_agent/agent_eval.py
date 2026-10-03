import json
import re
from pathlib import Path

from statbank_rag_agent.statbank import fetch_data
from statbank_rag_agent.tools import _normalize, _tableinfo

TOLERANCE = (
    0.01  # 1%: different tables can differ slightly in definitions or reference dates
)
NUMBER_PATTERN = re.compile(r"\d[\d\s.,]*\d|\d")
REFUSAL_PHRASES = (
    "could not",
    "couldn't",
    "cannot find",
    "can't find",
    "not find",
    "no table",
    "not available",
    "no data",
    "does not contain",
    "doesn't contain",
    "not covered",
    "unable to",
    "no statistics",
    "not possible",
    "does not provide",
    "doesn't provide",
    "does not have",
    "doesn't have",
    "does not publish",
    "doesn't publish",
    "no information",
    "not include",
    "there is no",
    "there are no",
)


def load_questions(path: str = "eval/agent.jsonl") -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Building the truth from label-based definitions ---


def resolve(truth: dict) -> tuple[str, dict[str, list[str]]]:
    """Turn {"table": ..., "select": {variable label: [value labels]}} into a table id and codes."""
    info = _tableinfo(truth["table"])
    if info is None:
        raise ValueError(f"unknown table {truth['table']}")
    selections = {}
    for key, labels in truth["select"].items():
        variable = next(
            (
                v
                for v in info["variables"]
                if _normalize(v["text"]) == _normalize(key)
                or v["id"].lower() == key.lower()
            ),
            None,
        )
        if variable is None:
            names = [v["text"] for v in info["variables"]]
            raise ValueError(
                f"{truth['table']}: no variable '{key}'. Variables: {names}"
            )
        codes = []
        for label in labels:
            match = next(
                (
                    v
                    for v in variable["values"]
                    if _normalize(v["text"]) == _normalize(label)
                ),
                None,
            )
            if match is None:
                similar = [
                    v["text"]
                    for v in variable["values"]
                    if _normalize(label) in _normalize(v["text"])
                ][:6]
                raise ValueError(
                    f"{truth['table']}.{variable['text']}: no value '{label}'. Similar: {similar}"
                )
            codes.append(match["id"])
        selections[variable["id"]] = codes
    return info["id"], selections


def true_value(truth: dict) -> float:
    table_id, selections = resolve(truth)
    rows = fetch_data(table_id, selections)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one number, got {len(rows)} rows")
    return float(rows[0]["value"])


# --- Checking answers ---


def numbers_in(text: str) -> list[float]:
    """Numbers in an answer, read with either comma, point or any kind of space as thousands separator.

    Models often write thousands separators as non-breaking or narrow spaces (such as U+202F),
    so every whitespace character is removed, not only the ordinary space.
    """
    found = []
    for raw in NUMBER_PATTERN.findall(text):
        compact = re.sub(r"\s", "", raw)
        for candidate in {
            compact.replace(",", ""),
            compact.replace(".", "").replace(",", "."),
        }:
            try:
                found.append(float(candidate))
            except ValueError:
                continue
    return found


def is_correct(answer: str, truth_value: float, tolerance: float = TOLERANCE) -> bool:
    return any(
        abs(n - truth_value) <= tolerance * abs(truth_value) for n in numbers_in(answer)
    )


def is_cited(answer: str) -> bool:
    return "statbank.dk/" in answer.lower()


def is_refusal(answer: str) -> bool:
    # Models often use typographic apostrophes, so they are normalised before matching
    low = answer.lower().replace("\u2019", "'")
    return any(phrase in low for phrase in REFUSAL_PHRASES)

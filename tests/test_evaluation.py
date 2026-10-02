from statbank_rag_agent.evaluation import evaluate, first_relevant_rank

QUESTIONS = [
    {"id": "a", "type": "t", "question": "first", "relevant": ["X"]},
    {"id": "b", "type": "t", "question": "second", "relevant": ["Y"]},
    {"id": "c", "type": "t", "question": "third", "relevant": ["Z"]},
]
FAKE_RESULTS = {
    "first": ["X", "A", "B"],  # correct at rank 1
    "second": ["A", "Y", "B"],  # correct at rank 2
    "third": ["A", "B", "C"],  # not found
}


def fake_search(question, k):
    return FAKE_RESULTS[question][:k]


def test_first_relevant_rank():
    assert first_relevant_rank(["A", "B", "X"], ["X", "Y"]) == 3
    assert first_relevant_rank(["A", "B"], ["X"]) is None


def test_metrics():
    result = evaluate(fake_search, QUESTIONS, k=3)
    assert result["hit@1"] == 1 / 3
    assert result["hit@5"] == 2 / 3
    assert abs(result["mrr"] - (1 + 0.5) / 3) < 1e-9

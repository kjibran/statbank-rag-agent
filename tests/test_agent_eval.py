from statbank_rag_agent.agent_eval import is_cited, is_correct, is_refusal, numbers_in


def test_numbers_with_different_thousands_separators():
    for text in ["378 361 people", "378,361 people", "378.361 people"]:
        assert 378361.0 in numbers_in(text), text


def test_correct_within_one_percent():
    assert is_correct("About 4 281 275 people lived there.", 4_281_275)
    assert is_correct("4,300,000 people", 4_281_275)  # within 1%
    assert not is_correct("4,500,000 people", 4_281_275)


def test_citation_and_refusal_detection():
    assert is_cited("Source: Statistics Denmark, StatBank.dk/folk1a")
    assert not is_cited("No source given.")
    assert is_refusal("I could not find a table about cats.")
    assert not is_refusal("378 361 people live in Aarhus.")

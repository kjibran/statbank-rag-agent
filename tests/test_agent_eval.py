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


def test_numbers_with_non_breaking_and_narrow_spaces():
    for space in [
        "\u00a0",
        "\u202f",
        "\u2009",
    ]:  # no-break, narrow no-break, thin space
        assert 349983.0 in numbers_in(f"349{space}983 people"), repr(space)


def test_real_refusals_from_the_agent_are_recognised():
    assert is_refusal(
        "I\u2019m sorry, but Statistics Denmark does not provide a table with the number of cats in Copenhagen."
    )
    assert is_refusal(
        "Statistics Denmark doesn\u2019t have data on unemployment in 1850."
    )


def test_adjacent_numbers_are_also_read_separately():
    numbers = numbers_in(
        "547 100\u2011year\u2011olds lived in Denmark on 1 January 2025."
    )
    assert 547.0 in numbers
    assert 349983.0 in numbers_in("349 983 people")  # the thousands reading still works

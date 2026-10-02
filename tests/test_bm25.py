from statbank_rag_agent.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords_and_plural_s():
    assert tokenize("How many farms produce milk?") == ["farm", "produce", "milk"]


def test_rare_word_outweighs_common_word():
    docs = [
        tokenize("Production and use of milk. Unit: Number."),
        tokenize("Number of cattle. Unit: Number."),
        tokenize("Number of pigs. Unit: Number."),
        tokenize("Number of farms. Unit: Number."),
    ]
    scores = BM25(docs).scores(tokenize("number milk"))
    assert scores.index(max(scores)) == 0  # "milk" is rare, "number" is everywhere

import math
import re
from collections import Counter

STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "did",
    "do",
    "does",
    "each",
    "for",
    "from",
    "get",
    "give",
    "got",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "many",
    "much",
    "my",
    "of",
    "on",
    "or",
    "our",
    "per",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, with a light plural strip ("farms" -> "farm")."""
    tokens = []
    for word in re.findall(r"[a-zæøå0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


class BM25:
    """Okapi BM25 ranking over a fixed set of documents.

    Rare words weigh more than common ones (inverse document frequency), repeated
    matches count with diminishing returns (k1), and long documents are not
    favoured just for being long (b).
    """

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.lengths) / len(documents)
        n = len(documents)
        document_frequency = Counter(term for doc in documents for term in set(doc))
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        result = []
        for counts, length in zip(self.term_counts, self.lengths):
            score = 0.0
            for term in query:
                tf = counts.get(term, 0)
                if tf:
                    norm = 1 - self.b + self.b * length / self.avg_length
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            result.append(score)
        return result

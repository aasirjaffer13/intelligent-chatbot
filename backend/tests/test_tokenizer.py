from __future__ import annotations

from app.nlp.tokenizer import naive_word_tokenize, sentence_tokenize, word_tokenize


class TestSentenceTokenize:
    def test_splits_multiple_sentences(self) -> None:
        result = sentence_tokenize("Hello world. How are you? Fine, thanks!")

        assert result == ["Hello world.", "How are you?", "Fine, thanks!"]

    def test_single_sentence_without_terminator(self) -> None:
        assert sentence_tokenize("just one clause") == ["just one clause"]

    def test_empty_and_whitespace_return_no_sentences(self) -> None:
        assert sentence_tokenize("") == []
        assert sentence_tokenize("   \n\t  ") == []

    def test_no_empty_fragments(self) -> None:
        result = sentence_tokenize("Hi.   There.  ")
        assert all(fragment for fragment in result)


class TestWordTokenize:
    def test_splits_words_and_punctuation(self) -> None:
        assert word_tokenize("Hello, world!") == ["Hello", ",", "world", "!"]

    def test_splits_contractions(self) -> None:
        # Treebank rules: "don't" -> ["do", "n't"] — a classic example of why
        # tokenizers are more than text.split().
        assert word_tokenize("I don't know.") == ["I", "do", "n't", "know", "."]

    def test_empty_input(self) -> None:
        assert word_tokenize("") == []
        assert word_tokenize("  ") == []

    def test_preserves_case(self) -> None:
        assert word_tokenize("NOVA Rules") == ["NOVA", "Rules"]


class TestNaiveTokenize:
    """The teaching implementation — and why we don't ship it."""

    def test_lowercases_and_drops_punctuation(self) -> None:
        assert naive_word_tokenize("Hello, world!") == ["hello", "world"]

    def test_keeps_contractions_whole(self) -> None:
        # Unlike Treebank, the naive version never splits contractions.
        assert naive_word_tokenize("I don't know") == ["i", "don't", "know"]

    def test_differs_from_nltk_on_contractions(self) -> None:
        assert naive_word_tokenize("can't") != word_tokenize("Can't")

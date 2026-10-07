from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.nlp.preprocessing import (
    PreprocessConfig,
    clean_text,
    lemmatize_tokens,
    preprocess,
    remove_stopwords,
    stem_tokens,
    strip_punctuation,
)


class TestCleanText:
    def test_full_cleaning(self) -> None:
        text = "Visit <b>https://example.com</b> NOW\tplease\nfor info"
        config = PreprocessConfig()

        assert clean_text(text, config) == "visit now please for info"

    def test_urls_removed(self) -> None:
        assert clean_text("see www.example.com now", PreprocessConfig()) == "see now"

    def test_urls_kept_when_disabled(self) -> None:
        config = PreprocessConfig(remove_urls=False, remove_html=False)
        assert "example.com" in clean_text("see www.example.com", config)

    def test_html_removed(self) -> None:
        assert clean_text("<p>Hello</p> <b>there</b>", PreprocessConfig()) == "hello there"

    def test_lowercase_can_be_disabled(self) -> None:
        config = PreprocessConfig(lowercase=False)
        assert clean_text("Hello World", config) == "Hello World"

    def test_whitespace_collapsed(self) -> None:
        assert clean_text("too    many\nspaces", PreprocessConfig()) == "too many spaces"


class TestStripPunctuation:
    def test_drops_punctuation_only_tokens(self) -> None:
        tokens = ["hello", ",", "!", "--", "world", "."]
        assert strip_punctuation(tokens) == ["hello", "world"]

    def test_keeps_tokens_with_letters(self) -> None:
        # "n't" and "'s" contain punctuation but carry meaning.
        assert strip_punctuation(["do", "n't", "'s", "e-mail"]) == [
            "do",
            "n't",
            "'s",
            "e-mail",
        ]

    def test_handles_empty_tokens(self) -> None:
        assert strip_punctuation(["", "hi"]) == ["hi"]


class TestStopwords:
    def test_removes_function_words(self) -> None:
        assert remove_stopwords(["the", "cat", "and", "the", "dog"]) == ["cat", "dog"]

    def test_english_negation_is_a_stopword(self) -> None:
        # Documented caveat: NLTK's list includes "not" — removing it can
        # invert meaning for sentiment-style tasks (see docs/nlp/01_preprocessing.md).
        assert remove_stopwords(["not", "happy"]) == ["happy"]


class TestStemming:
    def test_porter_rules(self) -> None:
        # Note how aggressive Porter is: "generously" -> "gener" (over-stemming,
        # a documented limitation) while lemmatization would be gentler.
        assert stem_tokens(["studies", "running", "generously", "cats"]) == [
            "studi",
            "run",
            "gener",
            "cat",
        ]

    def test_empty_list(self) -> None:
        assert stem_tokens([]) == []


class TestLemmatization:
    def test_noun_reduction(self) -> None:
        assert lemmatize_tokens(["studies", "cats"]) == ["study", "cat"]

    def test_verbs_need_pos_hint_limitation(self) -> None:
        # Documented limitation: default POS is noun, so verbs pass through.
        assert lemmatize_tokens(["running"]) == ["running"]


class TestPreprocessConfig:
    def test_stemming_and_lemmatization_are_exclusive(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessConfig(stemming=True, lemmatization=True)

    def test_defaults(self) -> None:
        config = PreprocessConfig()
        assert config.lowercase and config.remove_stopwords and config.remove_punctuation
        assert not config.stemming and not config.lemmatization


class TestPreprocessPipeline:
    def test_default_pipeline(self) -> None:
        result = preprocess("The Cats are sleeping! Visit https://x.com for more.")

        assert result.normalized_text == "the cats are sleeping! visit for more."
        assert result.sentences == ["the cats are sleeping!", "visit for more."]
        assert result.tokens == ["cats", "sleeping", "visit"]
        assert result.removed_stopwords == ["the", "are", "for", "more"]
        assert result.original_text.startswith("The Cats")

    def test_stopword_removal_can_be_disabled(self) -> None:
        result = preprocess("the cat", PreprocessConfig(remove_stopwords=False))
        assert result.tokens == ["the", "cat"]
        assert result.removed_stopwords == []

    def test_stemming_path(self) -> None:
        config = PreprocessConfig(remove_stopwords=False, stemming=True)
        assert preprocess("studies", config).tokens == ["studi"]

    def test_lemmatization_path(self) -> None:
        config = PreprocessConfig(remove_stopwords=False, lemmatization=True)
        assert preprocess("cats", config).tokens == ["cat"]

    def test_case_preserved_when_disabled(self) -> None:
        config = PreprocessConfig(lowercase=False, remove_stopwords=False)
        assert preprocess("Hello World", config).tokens == ["Hello", "World"]

    def test_empty_input(self) -> None:
        result = preprocess("")
        assert result.normalized_text == ""
        assert result.sentences == []
        assert result.tokens == []

    def test_punctuation_only_input_yields_no_tokens(self) -> None:
        assert preprocess("!!! ??? ...").tokens == []

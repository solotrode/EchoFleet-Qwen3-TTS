import pytest

from utils.text_chunker import chunk_text


def test_empty_text():
    assert chunk_text("") == []


def test_short_text_single_chunk():
    text = "The rich pay less in taxes than you do. And it's completely legal."
    chunks = chunk_text(text, max_chars=1000)
    assert len(chunks) == 1
    assert text.replace("  ", " ") in chunks[0]


def test_multiple_chunks_by_sentence():
    # create many short sentences to force multiple chunks
    sentence = "This is a short sentence."
    text = " ".join([sentence] * 100)
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    # ensure every chunk ends with a sentence ending punctuation
    for c in chunks:
        assert c.strip()[-1] in ".!?"


def test_sentence_longer_than_limit_keeps_whole_sentence():
    long_sentence = "A" * 500 + "."
    # max_chars smaller than sentence length, but chunker should keep the sentence whole
    chunks = chunk_text(long_sentence, max_chars=200)
    assert len(chunks) == 1
    assert chunks[0].endswith(".")

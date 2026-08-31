from defence_language_classifier.chunking import chunk_text, word_count


def test_chunks_respect_bounds():
    sentence = "This is a complete sentence containing enough words for deterministic testing. "
    chunks = chunk_text(sentence * 30, min_words=50, max_words=100)
    assert chunks
    assert all(50 <= word_count(chunk) <= 100 for chunk in chunks)


def test_short_document_remains_single_chunk():
    text = "This short document contains fewer words than the configured minimum."
    assert chunk_text(text, min_words=50, max_words=100) == [text]


def test_long_sentence_is_split_at_word_limit():
    text = " ".join(f"word{i}" for i in range(220))
    chunks = chunk_text(text, min_words=20, max_words=100)
    assert [word_count(chunk) for chunk in chunks] == [100, 100, 20]


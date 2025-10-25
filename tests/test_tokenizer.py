#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.tokenizer import CharacterTokenizer, BPETokenizer, get_tokenizer


class TestCharacterTokenizer:
    def test_initialization(self):
        tokenizer = CharacterTokenizer()
        assert tokenizer.vocab_size == 0

        tokenizer = CharacterTokenizer("abc")
        assert tokenizer.vocab_size == 3

    def test_train(self):
        tokenizer = CharacterTokenizer()
        text = "hello world"
        tokenizer.train(text)

        assert tokenizer.vocab_size > 0
        assert 'h' in tokenizer.stoi
        assert 'e' in tokenizer.stoi

    def test_encode_decode(self):
        tokenizer = CharacterTokenizer()
        text = "hello world"
        tokenizer.train(text)

        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)

        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_save_load(self, tmp_path):
        tokenizer = CharacterTokenizer()
        text = "test text"
        tokenizer.train(text)

        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(save_path))

        loaded_tokenizer = CharacterTokenizer.load(str(save_path))
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert loaded_tokenizer.stoi == tokenizer.stoi


class TestBPETokenizer:
    def test_initialization(self):
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size > 0

    def test_encode_decode(self):
        tokenizer = BPETokenizer()
        text = "Hello, world!"

        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)

        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_batch_operations(self):
        tokenizer = BPETokenizer()
        texts = ["Hello", "World", "Test"]

        encoded_batch = tokenizer.encode_batch(texts)
        assert len(encoded_batch) == 3
        assert all(isinstance(enc, list) for enc in encoded_batch)

        decoded_batch = tokenizer.decode_batch(encoded_batch)
        assert decoded_batch == texts


class TestTokenizerFactory:
    def test_get_character_tokenizer(self):
        tokenizer = get_tokenizer("character")
        assert isinstance(tokenizer, CharacterTokenizer)

    def test_get_bpe_tokenizer(self):
        tokenizer = get_tokenizer("bpe")
        assert isinstance(tokenizer, BPETokenizer)

    def test_invalid_tokenizer_type(self):
        with pytest.raises(ValueError):
            get_tokenizer("invalid_type")


if __name__ == "__main__":
    import tempfile

    # Test Character Tokenizer
    test_char = TestCharacterTokenizer()
    test_char.test_initialization()
    print("✓ Character tokenizer initialization test passed")
    test_char.test_train()
    print("✓ Character tokenizer training test passed")
    test_char.test_encode_decode()
    print("✓ Character tokenizer encode/decode test passed")

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        test_char.test_save_load(Path(tmpdir))
        print("✓ Character tokenizer save/load test passed")

    # Test BPE Tokenizer
    test_bpe = TestBPETokenizer()
    test_bpe.test_initialization()
    print("✓ BPE tokenizer initialization test passed")
    test_bpe.test_encode_decode()
    print("✓ BPE tokenizer encode/decode test passed")
    test_bpe.test_batch_operations()
    print("✓ BPE tokenizer batch operations test passed")

    # Test Factory
    test_factory = TestTokenizerFactory()
    test_factory.test_get_character_tokenizer()
    print("✓ Get character tokenizer test passed")
    test_factory.test_get_bpe_tokenizer()
    print("✓ Get BPE tokenizer test passed")

    print("\nAll tokenizer tests passed!")
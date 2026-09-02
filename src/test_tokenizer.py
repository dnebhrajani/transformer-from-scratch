from dataset import (
    load_raw_data,
    train_val_test_split,
    segment_pairs,
    get_or_train_tokenizers,
    test_tokenizer_roundtrip,
)


ciphers, plains = load_raw_data()
splits = train_val_test_split(ciphers, plains)

# Use the same 128-byte segmentation as your C1-C4 config
train_ciphers, train_plains = segment_pairs(
    splits["train"][0],
    splits["train"][1],
    segment_bytes=128,
)

# Train/load the tokenizers
src_tokenizer, tgt_tokenizer = get_or_train_tokenizers(
    train_ciphers,
    train_plains,
    src_vocab_size=4000,
    tgt_vocab_size=4000,
)

# Run the actual test
test_tokenizer_roundtrip(
    train_ciphers,
    train_plains,
    src_tokenizer,
    tgt_tokenizer,
    segment_bytes=128,
)
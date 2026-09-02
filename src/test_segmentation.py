from dataset import load_raw_data, segment_pairs

ciphers, plains = load_raw_data()

cipher = ciphers[0]
plain = plains[0]

seg_ciphers, seg_plains = segment_pairs(
    [cipher], [plain], segment_bytes=128
)

print("Original plaintext length:", len(plain))
print("Original cipher bits:", len(cipher))
print("Number of segments:", len(seg_plains))

for i in range(len(seg_plains)):
    start = i * 128
    end = min(start + 128, len(plain))

    expected_plain = plain[start:end]
    expected_cipher = cipher[start * 8:end * 8]

    print(f"\nSegment {i}:")
    print("  Plain length:", len(seg_plains[i]))
    print("  Cipher bits:", len(seg_ciphers[i]))
    print("  Plain aligned:", seg_plains[i] == expected_plain)
    print("  Cipher aligned:", seg_ciphers[i] == expected_cipher)
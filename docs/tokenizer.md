# Tokenizer
vocab:
the dictionary mapping every token string to its ID.
"hello" -> 14990
"!" -> 0

merges:
Rules for how to combine smaller token into bigger ones
merge 0: "i" + "n" -> "in"
BPE encoding starts with individual characters repeatedly applied the highest-priority merge
that matches.

addedtokens:
special tokens added on the top of the BPE vocabulary


# encode
  Step 1: Pre-tokenize — split text into words

  Split by whitespace, keeping the space as a Ġ prefix on the next word:

  "Hello world"  →  ["Hello", "Ġworld"]
  "I am fine"    →  ["I", "Ġam", "Ġfine"]

  Step 2: BPE — merge characters using merge rules

  For each word, start with individual characters, then repeatedly merge the highest-priority pair:

  "Ġworld" → ['Ġ', 'w', 'o', 'r', 'l', 'd']

  Scan all adjacent pairs, find the one with lowest merge index:
    ('Ġ','w') → merge index 500
    ('w','o') → merge index 300   ← lowest, apply this
    ('o','r') → merge index 800
    ('r','l') → merge index 1200
    ('l','d') → merge index 900

  → ['Ġ', 'wo', 'r', 'l', 'd']

  Repeat until no more merges apply:
  → ['Ġ', 'wor', 'l', 'd']
  → ['Ġ', 'world']
  → ['Ġworld']

  Step 3: Lookup — convert tokens to IDs

  ['Hello', 'Ġworld'] → [9707, 1879]


# BPE
Byte Pair Encoding - a way to build a vocabulary by repeatedly merging the most common pair
of characters.

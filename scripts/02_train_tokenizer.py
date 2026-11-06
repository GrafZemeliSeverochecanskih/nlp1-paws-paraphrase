"""Step 2: train and save the BPE tokenizer, then show example segmentations."""

from paws.config import Config
from paws.data import load_split, train_sentences
from paws.tokenization import train_and_save

cfg = Config()
corpus = train_sentences(cfg)
tokenizer = train_and_save(corpus, cfg)
print(f"Trained on {len(corpus)} distinct train sentences; vocabulary size {tokenizer.get_vocab_size()}")
first = load_split(cfg, "train")[0]
for sentence in (first["sentence1"], first["sentence2"]):
    print(tokenizer.encode(sentence).tokens)

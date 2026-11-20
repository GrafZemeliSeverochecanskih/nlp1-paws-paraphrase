# Blog post ideas (~1500 words, week 7)

Working title: "Same words, different meaning"

## Findings worth telling, per project
- P1:
- P2: An encoder that reads word order is not enough; the model must also compare the two sentences.
  Mean pooling and every cross-encoded LSTM stay at the majority baseline (0.56). Encoding each sentence
  separately and comparing them (siamese, [|u-v|, u*v]) jumps to 0.72 (LSTM) and 0.74 (BiLSTM). Learning
  curve: only the siamese LSTM improves with more data. Fine-tuning GloVe overfits after one epoch.
- P3:

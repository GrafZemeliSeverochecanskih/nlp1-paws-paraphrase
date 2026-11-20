# P1 error analysis

## Error categories
<!-- e.g. word swap, entity swap, negation, clause reordering, label noise -->

## Examples where bigrams fixed the unigram model

## Examples no model got right (-> motivation for P2)

# P2 error analysis

## Same-word-set pairs (the main research question)
Both sentences use the same word set in 13% of the pairs (64% of them labelled *different*). Mean pooling gives the
two sentences cosine 1.000 there (`notebooks/p2_embeddings.ipynb`), so a pooling system cannot separate them.
Accuracy over 3 seeds x 5 grouped folds (`results/p2_seeds0-1-2/same_word_set_table.csv`):

| system | same word set | other pairs |
|---|---|---|
| majority label within group | 0.637 | 0.548 |
| lstm+siamese | 0.745 | 0.714 |
| bilstm+siamese | 0.763 | 0.735 |
| bilstm+siamese, GloVe 2014 300d | 0.766 | 0.748 |

The siamese LSTMs beat the group's majority label by 11-13 points on pairs where only word order differs.

## Bayesian feature analysis (`scripts/11`, 4,000 pairs, gold label as control)
`results/p2/bda/bda_summary.csv`. Posterior mean weights on standardised features (positive = more often correct);
89% interval in brackets where it excludes 0.

| feature | pool+cross | lstm+cross | lstm+siamese | bilstm+siamese |
|---|---|---|---|---|
| is_paraphrase (control) | -2.81 | -2.28 | -0.13 | -0.26 |
| word_overlap | ~0 | ~0 | +0.09 | +0.17 [0.09, 0.24] |
| length | ~0 | ~0 | -0.09 | -0.07 |
| num_capitalised | ~0 | +0.22 | +0.10 | +0.07 |
| same_word_set | ~0 | ~0 | ~0 | -0.09 |
| order_change, oov_rate | ~0 | ~0 | ~0 | ~0 |

- Without the label control, `order_change` looked strongly predictive for the cross systems (+1.0). That was the
  label: those systems predict *different* almost always, and reordered pairs are mostly *different*. With the
  control, the cross systems' correctness is explained by the label (all other intervals include 0, except
  num_capitalised for lstm+cross, +0.22 [0.12, 0.32]); their errors hardly depend on the input.
- For the siamese systems, order_change and oov_rate (and same_word_set for the LSTM) include 0; the other weights
  exclude 0 but are small (|w| <= 0.26) next to the cross systems' label weight.
- The siamese systems are slightly worse on paraphrases and, for the BiLSTM, on same-word-set pairs; better on
  high-overlap pairs and worse on long pairs (one LSTM state per sentence). Order change and OOV rate do not
  predict their errors.
- Limitation: features are computed on the pair only; correctness is from one seed's out-of-fold predictions.

## Error categories (pairs both siamese LSTMs get wrong: 3,350 of 20,000, seed 0)
From `results/p2_seeds0-1-2/error_examples.md` (10 sampled pairs, so the categories are indicative, not counted):
- **Paraphrases with a moved clause plus light rewording**, predicted *different*: "During the American Civil War,
  Indiana was ..." / "Indiana was ... during the American Civil War"; "met in Sydney on 31 August" / "arrived ...".
- **Role swaps of unknown names**: "hosted by Guy Zu-Aretz and Eli Mizrahi with judges Yarden Harel ..." vs the
  swapped version; all OOV names map to `<unk>`, so the swap is invisible.
- **Role swaps of two common words**: "free charges ... total charge" vs "total fees ... free charge".
- **Label noise / ungrammatical back-translations**: "Clayton Mendel Stromm produces", "It is in its rare form
  unhybridized", and a *paraphrase* label on a District/Region swap.

## Examples where the siamese LSTM fixed pooling (lstm+siamese right, pool+cross wrong: 5,893 of 20,000, seed 0)
From `results/p2/error_examples.md`. pool+cross predicts *different* almost always, so most of these are paraphrases
it misses (8 of the 10 sampled); the informative ones are the *different* pairs the LSTM gets right:
- **Swapped modifiers**: "a *different* organisation, a scout group, combines *local* sections" vs "a *local*
  organisation ... combines *different* sections". Nearly the same words, so pooling sees almost the same vector; the LSTM tracks
  which noun each adjective attaches to.
- **Swapped verb arguments**: "mosques were bombed and Hui was slaughtered or destroyed" vs "Mosques were destroyed
  and ... Hui were slaughtered ... or bombed".
- **Paraphrases with moved phrases**: "In the evening, finally a bowl came ..." vs "Finally, in the evening, came a
  bowl ..." (same word set, label *paraphrase*): the LSTM accepts the reordering that pooling cannot even see.

Only 560 of 20,000 pairs are wrong for all 17 P2 systems.

## What to prioritise next
1. Cross-sentence interaction beyond [|u-v|, u*v]: attention between the two sentences (P3 transformers).
2. Character or subword input for the OOV names PAWS swaps.
3. A cleaned evaluation subset, since part of the remaining errors look like label noise.

# Brain-to-Text Upgrade Plan (Contextual Normalization + Confidence-Gated LM)

## Goal
Improve decoding accuracy before moving to the 5‑gram LM by boosting preprocessing quality:

1. Normalize neural features across days & within sessions using a contextual transformer.
2. Calibrate phoneme logits and attach confidence-aware filtering before the LM.
3. Enrich the n‑gram LM with articulatory-aware tokens and dynamic weighting.
4. Optionally add DistilBERT rescoring for innovation/robustness.

## Current State
- Decoder: 5-layer GRU (baseline) with day-specific linear layers.
- Preprocessing: Gaussian smoothing + minor jitter; no phoneme-level filtering.
- LM: 1/3-gram baseline with OPT rescoring; 5-gram planned.

## Proposed Components

### 1. Contextual Neural Normalizer
- **Placement:** Before GRU in `rnn_model.py`.
- **Architecture:** 1–2 transformer encoder layers (hidden 512, 8 heads), taking neural features + embeddings for day/block.
- **Purpose:** Align neural distributions across days and intra-day state shifts, reducing load on day-specific matrices.
- **Implementation Steps:**
  1. Create `contextual_normalizer.py` module.
  2. Feed normalized output into existing day-specific linear layers (or replace them entirely).
  3. Train end-to-end with baseline loss; monitor cross-day validation PER improvements.

### 2. Phoneme Confidence Gate + Articulatory Tokens
- **Confidence Head:** Bidirectional GRU (128 units) or 1D CNN on decoder logits to output smoothed logits + confidence mask.
- **Filtering:** Drop or down-weight frames with entropy above threshold before sending to LM.
- **Token Augmentation:** Map phonemes to articulatory classes (vowel, plosive, etc.) and pass tokens as `PHONEME|CLASS`.
- **Dynamic LM Mixing:** Use confidence statistics to adjust `acoustic_scale`, `blank_penalty`, `alpha` per utterance via a 2-layer MLP.
- **Implementation Steps:**
  1. Extend `evaluate_model_helpers.py` to compute confidences and add class tags.
  2. Update LM training scripts to expect tokens with class suffix.
  3. Log confidence stats for analytics.

### 3. Enhanced n-gram LM + External Text
- **Data:** Phonemize provided transcripts + public corpora (conversational + medical text) using CMUdict or g2p.
- **Build:** Train 5-gram LM (SRILM/OpenFST). Store under `language_model/pretrained_language_models/openwebtext_5gram_lm_sil`.
- **OPT Parameters:** Keep `--rescore --do_opt` for best accuracy; adjust dynamic weights from Step 2.

### 4. DistilBERT Rescoring (Optional but innovative)
- **Input:** n-best candidate sentences + phoneme confidence summary.
- **Model:** DistilBERT fine-tuned to rank true sentences higher (supervised on validation data).
- **Integration:** After OPT output, run DistilBERT to select final sentence before CSV dump.

## Analytics Loop
1. Instrument `evaluate_model.py` to dump per-phoneme confusion matrices and entropy statistics.
2. Visualize errors (vowels vs consonants) to guide calibration thresholds.
3. Use analytics to validate improvements after each component is added.

## Execution Order
1. **Analytics + Logging:** add phoneme stats, run baseline to capture current issues.
2. **Contextual Normalizer:** implement transformer module, retrain GRU.
3. **Confidence Gate + Tokens:** add calibration head, articulatory tags, dynamic LM weights.
4. **Enhanced 5-gram LM:** train with external text, integrate.
5. **DistilBERT Rescoring:** if time allows, train ranking model on validation n-best outputs.

## Infrastructure Notes
- Training/eval will run on Vast.ai (e.g., 4× RTX 3090, 413 GB RAM) so transformer+confidence modules can be moderately sized.
- Keep code modular to scale down later if on-device constraints appear.

## Next Steps
1. Implement analytics instrumentation and run baseline to gather phoneme errors.
2. Start coding contextual normalizer module and integrate into `rnn_model.py`.
3. Design articulatory mapping table and confidence head scaffolding.



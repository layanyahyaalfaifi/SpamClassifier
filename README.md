# Spam Classifier from Text & Audio

A multimodal spam detection system that classifies emails as spam or not spam — from both typed text and voice input. Built with Naïve Bayes and OpenAI Whisper for speech-to-text transcription.

🔗 **Live Demo:** [spam-audio-app on Streamlit](https://spam-audio-app-ejqhhtygiqjjxaw8pswt4d.streamlit.app/)

---

## How It Works

1. User inputs an email as text or uploads an audio clip (MP3, WAV, M4A)
2. Audio is transcribed to text using **OpenAI Whisper**
3. Text is cleaned, tokenized, and vectorized with **TF-IDF**
4. **Naïve Bayes** classifier predicts: `SPAM` or `NOT SPAM`

---

## Results

### Naïve Bayes (primary model)

| Metric | Not Spam | Spam | Overall |
|---|---|---|---|
| Accuracy | — | — | **99%** |
| Precision | 0.99 | 0.99 | 0.99 |
| Recall | 1.00 | 0.97 | 0.98 |
| F1-score | 0.99 | 0.98 | 0.99 |

### Random Forest (baseline comparison)

| Metric | Not Spam | Spam | Overall |
|---|---|---|---|
| Accuracy | — | — | **98%** |
| Precision | 0.97 | 1.00 | 0.99 |
| Recall | 1.00 | 0.91 | 0.96 |
| F1-score | 0.99 | 0.95 | 0.97 |

**Why Naïve Bayes over Random Forest?** Random Forest had perfect spam precision (1.00) but missed 9% of actual spam (recall 0.91). Naïve Bayes had a better spam recall (0.97), meaning fewer spam emails slipping through — which is what matters most in a spam filter. It also runs significantly faster, making it better for real-time classification.

### Inference Latency

| Input Type | Expected | Actual |
|---|---|---|
| Text | ~3s | ~2.12–2.43s |
| Audio (via Whisper) | ~3s | ~2.39–2.40s |

---

## Dataset

**Spam Email Dataset** — [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)

| Stat | Value |
|---|---|
| Total emails | 5,728 |
| Spam | 2,300 (40%) |
| Not spam | 3,428 (60%) |
| Language | English |

---

## Model Details

- **Algorithm:** Multinomial Naïve Bayes
- **Vectorization:** TF-IDF (`max_features=3000`)
- **Smoothing:** Laplace (α = 0.03) — chosen over default α=1 to preserve distinction between frequent and rare words
- **Speech-to-text:** OpenAI Whisper (`medium` model)

---

## Limitations

- **English-only** — no support for Arabic or multilingual content
- **Whisper requires paid API** for deployment; free platforms struggle with its resource demands
- **Noisy audio** reduces transcription accuracy, which can affect classification
- **No feedback loop** — users can't flag wrong predictions to improve the model over time
- **Deployment constraints** — free hosting introduces latency and storage limits

---

## Future Work

- Add Arabic and multilingual support
- Replace Naïve Bayes with BERT for higher accuracy on complex patterns
- Integrate a user feedback mechanism to improve predictions over time
- Analyze email attachments and embedded links for deeper spam detection
- Deploy with FastAPI for better performance and scalability

---

## Setup

```bash
pip install pandas scikit-learn nltk matplotlib seaborn openai-whisper
sudo apt install ffmpeg
jupyter notebook spam_classifier.ipynb
```

---

## References

- Rahman et al., [Spam Email Detection using Naïve Bayes](https://www.itm-conferences.org/articles/itmconf/pdf/2025/01/itmconf_dai2024_04028.pdf), 2025
- Islam et al., [Evaluating ML Methods for Spam Detection](https://www.sciencedirect.com/science/article/pii/S1877050921013016), Procedia CS 2021

# 🟢 Sentira — Customer Sentiment Analysis System

> An end-to-end NLP + Machine Learning desktop application for analysing Amazon product reviews, built with Python and a custom Positivus-inspired GUI.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Overview

**Sentira** is a desktop sentiment analysis tool that classifies Amazon product reviews as **Positive**, **Negative**, or **Neutral** using multiple machine learning models. It features a bold, modern GUI inspired by the Positivus design system — white, black, and electric lime green — with interactive charts, keyword extraction, scrollable pages, and a full ML pipeline.

This project was built as part of a **BS Artificial Intelligence** portfolio to demonstrate practical skills in NLP, supervised learning, and GUI development.

---

## ✨ Features

### 🧠 Machine Learning Pipeline
- **TF-IDF Vectorisation** — bigram features, top 2500 tokens
- **3 Trained Models** compared side-by-side:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Random Forest (100 estimators)
- **80/20 stratified train-test split**
- Accuracy scores + confusion matrix for best model

### 📊 Interactive Dashboard (4 Pages)

| Page | What It Shows |
|------|---------------|
| **Overview** | Sentiment by category (grouped bar), share of voice (donut), full 1–5 star rating distribution |
| **Keywords** | Top 10 most frequent words for each sentiment class (horizontal bar charts) |
| **ML Models** | Accuracy comparison across all 3 models + colour-coded confusion matrix |
| **Explore** | Searchable, scrollable table of all 800 reviews with live filtering |

### 🔍 Other Features
- **Predict Single Review** — type any custom text and get instant sentiment + confidence %
- **Load CSV** — upload your own Amazon review dataset
- **Export Results** — save processed data as CSV
- **Sentiment filter pills** — filter by Positive / Negative / Neutral before running analysis
- **Category filter** — drill down by Electronics, Beauty, Books, Sports, Home & Kitchen
- **Scrollable pages** — every page scrolls with mouse wheel, nothing is cut off
- **Synthetic dataset** — 800 realistic Amazon reviews auto-generated on launch

---

## 🖼️ GUI Design

The app uses a **Positivus-inspired design system**:

- 🟢 Electric lime green `#B9FF66` as primary accent
- ⬛ Near-black `#191A23` for cards and header
- ⬜ Pure white background for content areas
- Thick black bordered cards on every section
- Pill-style tag labels on all card headers
- Bold Arial Black typography for headings and numbers
- Red `#FF4F5A` for negative, amber `#FFA826` for neutral

---

## 🗂️ Project Structure

```
sentira/
│
├── sentiment_app.py        # Main application — run this
└── README.md               # This file
```

> No external data files needed — the app generates a synthetic 800-review dataset automatically on launch.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentira-sentiment-analysis.git
cd sentira-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```

> **Note:** `tkinter` comes pre-installed with Python on Windows. No extra install needed.

### 3. Run the app
```bash
python sentiment_app.py
```

---

## 🚀 How to Use

1. **Launch** — app opens and auto-loads 800 Amazon reviews
2. **Filter** *(optional)* — choose a category or sentiment using the pill buttons
3. **Click ▶ Analyse Reviews** — runs the full ML pipeline (takes ~3–5 seconds)
4. **Explore tabs** — switch between Overview, Keywords, Models, Explore
5. **Predict Review** — click the button in the top bar to test any custom review text
6. **Load CSV** — upload your own review dataset (needs a `review_text` column)
7. **Export** — save the current results to CSV

---

## 📁 CSV Format (for custom datasets)

Your CSV file must include at minimum:

| Column | Required | Description |
|--------|----------|-------------|
| `review_text` | ✅ Yes | The product review text |
| `rating` | Optional | Star rating 1–5 (auto-filled as 3 if missing) |
| `category` | Optional | Product category (auto-filled as "General") |
| `sentiment` | Optional | Pre-labelled sentiment (auto-detected if missing) |
| `helpful_votes` | Optional | Number of helpful votes (auto-filled as 0) |

Also accepted: `review`, `text`, `comment`, or `body` as column name — auto-renamed.

---

## 🧪 Model Performance (Synthetic Dataset)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~100% |
| Naive Bayes | ~96% |
| Random Forest | ~100% |

> ⚠️ High accuracy is expected on synthetic data since reviews follow consistent patterns. Real-world performance will vary.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| Tkinter | Desktop GUI framework |
| Scikit-learn | ML models + TF-IDF vectoriser |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | All charts and visualisations |
| re / Counter | Text cleaning + keyword extraction |

---

## 📚 NLP Pipeline Detail

```
Raw Text
   ↓
Lowercase + Remove special characters
   ↓
Remove stopwords (custom 100+ word list)
   ↓
TF-IDF Vectorisation (bigrams, max 2500 features)
   ↓
Train/Test Split (80/20, stratified)
   ↓
Train 3 ML Models in parallel
   ↓
Evaluate accuracy + generate confusion matrix
   ↓
Display results in GUI
```

**Rule-based fallback** is also used for single-review prediction (keyword matching with confidence scoring — no model training required).

---

## 👩‍💻 Author

**Eisha Khan**
BS Artificial Intelligence Student

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🔗 Related Projects

- 📈 [Sales Demand Forecasting System](../sales-forecasting) — FMCG demand prediction with Tkinter GUI
- 🎓 [University Advisor System](../university-advisor) — Intelligent university recommender with Streamlit
- 📐 [Gram-Schmidt Calculator](../gram-schmidt) — Linear algebra console tool

# Yelp Review Classifier

A text classification project that predicts whether a Yelp review is **1-star** or **5-star** from the review text alone, using classic NLP techniques and scikit-learn.

The project also compares two ways of turning text into features, **Bag-of-Words (CountVectorizer)** and **TF-IDF**, and explains why the simpler one won with Naive Bayes.

---

## Project Structure

```
Yelp_Review_Classifier/
├── yelp.csv             # Dataset (Yelp reviews)
├── data.py              # Loading, EDA, filtering, vectorization, train/test split
├── train.py             # Model training and evaluation
├── main.py              # Entry point: runs the full pipeline
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** Yelp review dataset (`yelp.csv`, ~10,000 reviews)
- **Key columns:** `text` (review), `stars` (1–5 rating), plus `cool`, `useful`, `funny` vote counts
- **Filtered to:** only 1-star and 5-star reviews, to frame the task as a clear binary classification of very negative vs very positive reviews

---

## Approach

### 1. Exploratory Data Analysis
- Added a `text length` feature (number of characters per review)
- Plotted the text length distribution across star ratings
- Grouped by `stars` and computed mean `cool`, `useful`, `funny` and `text length`
- Built a correlation matrix of these features

### 2. Preprocessing
- Kept only 1-star and 5-star reviews
- Removed English stopwords using NLTK
- Converted text into numeric features with:
  - **CountVectorizer** (raw word counts)
  - **TF-IDF** (term frequency × inverse document frequency)

### 3. Modeling
- **Multinomial Naive Bayes** trained on each feature type
- 70 / 30 train-test split (`random_state=101`)
- Evaluated with a confusion matrix and classification report (precision, recall, F1)

---

## Results

| Features         | Model          | Accuracy | 1-star F1 | 5-star F1 |
|------------------|----------------|----------|-----------|-----------|
| CountVectorizer  | Multinomial NB | _xx%_    | _x.xx_    | _x.xx_    |
| TF-IDF           | Multinomial NB | _xx%_    | _x.xx_    | _x.xx_    |

**Key finding:** adding TF-IDF made the Naive Bayes model *worse*.

### Why TF-IDF hurt performance
- **Multinomial NB expects count data.** It models how often each word appears. TF-IDF replaces counts with small fractional weights that no longer fit that assumption.
- **The classes are imbalanced.** There are many more 5-star than 1-star reviews. With TF-IDF features, the model leaned heavily toward predicting 5 stars, so recall on the 1-star class collapsed.
- **Accuracy alone hides this.** The per-class F1 scores and the confusion matrix make the drop clear.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/LovelaceLoops/Yelp_Review_Classifier.git
cd Yelp_Review_Classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

`main.py` is the single entry point. It imports `data.py` (preprocessing) and then `train.py` (training and evaluation), so the whole project runs with one command. NLTK stopwords are downloaded automatically on the first run.

You can also run each stage on its own:
```bash
python data.py    # EDA and preprocessing only
python train.py   # Training and evaluation
```

---

## Tech Stack

- **Python**
- **pandas, NumPy**: data handling
- **Matplotlib, Seaborn**: visualization
- **scikit-learn**: vectorization, Naive Bayes, evaluation
- **NLTK**: stopword removal

---

## Future Improvements

- Pair TF-IDF with **Logistic Regression** or **LinearSVC**, models that suit TF-IDF better than NB
- Handle class imbalance with `class_weight='balanced'` or resampling
- Add **bigrams** (`ngram_range=(1, 2)`) to capture phrases like "not good"
- Extend to all 5 star ratings (multi-class) or 3-class sentiment
- Try transformer models (e.g. DistilBERT) for comparison
- Save the trained model and vectorizer with `joblib` and serve predictions through a small Streamlit or Flask app

---

## What I Learned

- Debugging common pandas pitfalls (e.g. `groupby().mean()` on text columns in pandas 2.x, `df('col')` vs `df['col']`)
- Structuring an ML project as separate Python modules with a single `main.py` entry point
- That a more sophisticated feature (TF-IDF) is not automatically better, because the model and the feature representation have to fit each other
- Why per-class metrics matter more than accuracy on imbalanced data

---

## Author

**Mrunal Patil**: B.E. Computer Engineering (AI/ML), University of Mumbai
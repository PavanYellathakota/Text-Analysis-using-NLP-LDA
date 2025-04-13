# Advanced Text Analysis using NLP: Sentiment Analysis, Named Entity Recognition (NER) & Topic Modeling (LDA)
### Extracting emotional tone, entities, and topics from unstructured textual data

- This project analyzes a dataset of articles stored in `data/Text.csv` to uncover insights through various text analysis techniques: Word Cloud, Sentiment Analysis, Named Entity Recognition (NER), and Latent Dirichlet Allocation (LDA) Topic Modeling. The analysis was performed using Python in a Jupyter Notebook environment with libraries like `pandas`, `spacy`, `textblob`, `plotly`, `wordcloud`, and `scikit-learn`.

---

## Dataset Overview

The dataset (`data/Text.csv`) contains articles with titles and text content. A preview of the first few rows:

| Article (Excerpt)                                      | Title                                              |
|-------------------------------------------------------|----------------------------------------------------|
| Data analysis is the process of inspecting and...    | Best Books to Learn Data Analysis                 |
| The performance of a machine learning algorith...    | Assumptions of Machine Learning Algorithms        |
| You must have seen the news divided into categ...    | News Classification with Machine Learning         |
| When there are only two classes in a classific...    | Multiclass Classification Algorithms in Machine...|
| The Multinomial Naive Bayes is one of the vari...    | Multinomial Naive Bayes in Machine Learning       |

- **Encoding**: Loaded with `latin-1` encoding.
- **Size**: Contains 34 articles (based on sentiment analysis count).

---

## Analysis Techniques and Findings

### 1. Word Cloud
A word cloud was generated to visualize frequent terms in article titles.
![word cloud](assets/Results/wordcloud.png)

- **Key Words**: `recorder`, `frameworks`, `stock`, `apis`, `multinomial`, `values`, `resources`, `apple`, `deep`, `computer`, `networks`, `assumptions`, `bayes`, `prediction`, `analysis`, `best`, `health`, `applications`, `news`, `machine`, `learning`, `clustering`, `insurance`, `introduction`, `learn`, `classification`, `algorithm`, `python`, `data`, `sentiment`.
- **Insights**:
  - Dominant terms include `machine`, `learning`, `data`, `python`, and `algorithm`, reflecting a focus on machine learning and data science.
  - Specific terms like `multinomial`, `bayes`, and `clustering` suggest technical content.

---

### 2. Sentiment Analysis
Sentiment analysis was performed on the `Article` column using TextBlob, with polarity scores categorized as Positive (>0.1), Neutral (-0.1 to 0.1), or Negative (<-0.1).

- **Sentiment Summary**:
count    34.000000
mean      0.160138 (slightly positive)
std       0.296732
min      -0.800000
25%       0.000000
50%       0.134464
75%       0.332299
max       0.666667

- **Category Distribution**:
Positive    18 (52.9%)
Neutral     11 (32.4%)
Negative     5 (14.7%)

![Sentiments](assets/Results/Sentiment.png)

- **Visualization Insights** (Plotly Histogram with Rug Plot):
- **Positive (Green)**: Peaks in the 0.2 to 0.6 range, indicating many articles have a positive tone.
- **Neutral (Purple)**: Concentrated around 0, forming a significant portion.
- **Negative (Red)**: Fewer articles, mostly between -0.1 and -0.2, with rare outliers near -1.
- **Outliers**: Rug plot shows a few strongly positive (near 1) and negative (near -1) articles.
- **Conclusion**: The articles are predominantly positive or neutral, with a mean sentiment of 0.16, suggesting an optimistic or informative tone.

---

### 3. Named Entity Recognition (NER)
NER was applied using spaCy to identify and count named entities in the articles.

![Frequency](assets/Results/Frequency.png)

- **Top Individual Entities**:
- `today`: Most frequent, likely a temporal reference.
- `Machine Learning`: Second most frequent, highlighting its prominence.
- Others: `one` (twice), `September`, `Instagram`, `Pfizer`, `Apple`, `the K-Means`, `Data Scientist`.
- **Insights**:
- **Temporal/Numerical**: `today`, `September`, and `one` suggest time and quantity references.
- **Companies/Products**: `Instagram`, `Pfizer`, `Apple` indicate mentions of specific entities.
- **Technical Terms**: `Machine Learning`, `the K-Means`, `Data Scientist` point to data science and ML focus.
- **Frequency Trend**: Visualized with a 'Viridis' color scale (yellow for high frequency, purple for low), showing a steep drop-off after `today`.

---

### 4. LDA Topic Modeling
LDA was used to identify 5 topics from the preprocessed article text, with top words extracted for each.

![LDA](assets/Results/LDA.png)

- **Topics and Top Words**:
1. **Topic 1**: `learn, python, insurance, application, machine, want, algorithm, assumption, tutorial, science`
   - Focus: Learning machine learning with Python, possibly tutorials or assumptions.
2. **Topic 2**: `algorithm, python, learning, machine, classification, api, base, clustering, bayes, datum`
   - Focus: ML algorithms (classification, clustering, Bayes) and Python.
3. **Topic 3**: `analysis, sentiment, datum, want, learn, people, python, data, analyze, today`
   - Focus: Data and sentiment analysis with Python.
4. **Topic 4**: `news, category, computer, learning, machine, website, good, know, popular, task`
   - Focus: News categorization and ML applications.
5. **Topic 5**: `learning, machine, clustering, algorithm, stock, learn, language, deep, python, price`
   - Focus: Advanced ML (clustering, deep learning) and applications like stock prices.
- **Insights**:
- Overlap in terms like `machine`, `learning`, `python`, and `algorithm` across topics, indicating a cohesive ML theme.
- Diverse applications: insurance, news, sentiment, stock prices.
- Visualized with a 'Viridis' bar chart, showing top words per topic.

---

## Key Findings
- **Theme**: The articles center on machine learning and data science, with frequent mentions of Python, algorithms, and applications like classification and clustering.
- **Tone**: Mostly positive (52.9%) or neutral (32.4%), with a mean sentiment of 0.16.
- **Entities**: `today` and `Machine Learning` dominate, alongside companies (e.g., `Apple`) and technical terms (e.g., `Data Scientist`).
- **Topics**: Five distinct yet overlapping topics highlight learning, analysis, and practical ML applications.

---

## Methodology
- **Tools**: Python with `pandas`, `spacy`, `textblob`, `wordcloud`, `plotly`, `scikit-learn`.
- **Preprocessing**: Lemmatization, stopword/punctuation/digit removal for LDA; raw text for sentiment and NER.
- **Visualizations**: Word Cloud, Plotly histograms/bars with 'Viridis' color scale.

---

## Usage
1. **Setup**: Install dependencies (`pip install pandas spacy textblob wordcloud plotly scikit-learn`) and spaCy model (`python -m spacy download en_core_web_sm`).
2. **Run**: Execute in Jupyter Notebook with `data/Text.csv` in the `data/` directory.
3. **Explore**: Interact with Plotly charts (zoom, hover) and review printed outputs.

---

## Future Work
- **Expand Dataset**: Analyze more articles for broader insights.
- **Refine Topics**: Adjust LDA `n_components` or preprocessing for sharper topics.
- **Sentiment Granularity**: Explore subjectivity alongside polarity.
- **NER Enhancement**: Filter entity types (e.g., only PERSON, ORG) for specific focus.

---
  
## Author

<div align="center">
  <img src="assets/PYE.svg" alt="Author Banner" style="width:100%; height:auto; border-radius: 8px;">
</div>

**Author**: [Pavan Yellathakota]  
**Date**: FEB 2025  

---

## Contact Information

You can reach out to me through the following channels:

- **Email**: [pavanyellathakota@gmail.com](mailto:pavanyellathakota@gmail.com)
- **LinkedIn**: [Pavan Yellathakota](https://www.linkedin.com/in/pavanyellathakota/)

For more projects and resources, check out:

- **GitHub**: [Pavan Yellathakota](https://github.com/PavanYellathakota)
- **Portfolio**: [pye.pages.dev](https://pye.pages.dev)

---



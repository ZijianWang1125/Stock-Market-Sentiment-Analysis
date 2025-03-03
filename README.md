# Stock Market Sentiment Analysis

**Stock Market Sentiment Analysis** is a project that explores the relationship between online investor sentiment and stock market returns using advanced Natural Language Processing (NLP) techniques. In particular, the project leverages social media and forum posts to quantify investor sentiment and integrates this information into a predictive model for the Shanghai Composite Index.

---

## Table of Contents

- [Stock Market Sentiment Analysis](#stock-market-sentiment-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Background \& Motivation](#background--motivation)
  - [Objectives](#objectives)
  - [Project Workflow](#project-workflow)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Expected Results \& Analysis](#expected-results--analysis)
  - [Discussion \& Future Work](#discussion--future-work)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

---

## Overview

This project gathers labeled Chinese financial sentiment data, fine-tunes a BERT-based sentiment scoring model, and subsequently applies the model to extensive web-scraped forum posts from EastMoney-Shanghai Securities Composite index (SSEC) forum吧. The derived sentiment scores are then aggregated into a monthly sentiment index which is used as an additional factor in a regression model to predict the next-day returns of the Shanghai Composite Index.

---

## Background & Motivation

- **Investor Sentiment Impact:**
Investor emotions extracted from online posts significantly influence market behavior and can be key indicators of future market movements.

- **Shanghai Composite Index as a Barometer:**
The Shanghai Composite Index is used to represent overall market sentiment and economic health in China.

- **Research Goals:**
- Quantify investor sentiment using advanced NLP models.
- Integrate sentiment indices with traditional financial indicators.
- Leverage sentiment to enhance predictions of stock index returns and inform trading strategies.

---

## Objectives

1. **Data Acquisition:**

- Gather labeled Chinese financial sentiment datasets from sources such as Github, Kaggle, and Hugging Face.
- The sentiment labels are “Positive” (1), “Neutral” (0) and “Negative” (-1).

2. **Model Development:**

- Use a Chinese pre-trained BERT model.
- Augment the BERT model with a fully connected layer to output sentiment probability scores.
- Fine-tune the model for 10 epochs on the labeled dataset.

3. **Data Collection via Web Scraping:**

- Use Beautiful Soup to scrape posts and comments from EastMoney-Shanghai Securities Composite index (SSEC) forum吧 for the period from 2018/12/01 to 2025/01/01.
- Apply the fine-tuned BERT model to each post/comment to obtain a sentiment score.

4. **Sentiment Index Computation:**

- Employ a rolling window approach to compute a monthly sentiment index for the period 2019/01/01 to 2025/01/01.
- For each month \( t \) (e.g., \( t \in \{\text{2019-01-01}, \dots, \text{2025-01-01}\} \)), use all posts/comments in the period \([t-1, t]\) to calculate:
- For each comment \( i \):

 \[
 \text{score}_i = f_{\text{BERT}, \text{prob}}(comment_i)
 \]

- Positive sentiment sum:

 \[
 \text{pos}_t = \sum_{i=1}^{n_t} \text{score}_i
 \]

- Negative sentiment sum:

 \[
 \text{neg}_t = \sum_{i=1}^{n_t} \left(1-\text{score}_i\right)
 \]

- Sentiment index calculation:

 \[
 \text{index}_t = \ln\left(\frac{1+\text{pos}_t}{1+\text{neg}_t}\right)
 \]

 **Note:** We intentionally avoid using likes (denoted as \(\omega_i\)) for weighting the scores as they may originate from future periods and hence do not accurately capture the sentiment at time \( t \).

5. **Predictive Modeling:**

- Incorporate the sentiment index as an additional factor in the regression model to forecast the next-day return of the index:

 \[
 r_{t+1} = \alpha + \sum_{i=1}^{n} \beta_i \, \text{factor}_{i,t} + \beta_{n+1} \, \text{index}_t + \epsilon_t
 \]

 where:

 \[
 r_{t+1} = \frac{P_{t+1} - P_t}{P_t}
 \]

 represents the return of the Shanghai Composite Index.

6. **Investment Strategy & Backtesting:**

- Use the regression model to predict returns:

 \[
 \hat{r}_{t+1} = \alpha + \sum_{i=1}^{n} \beta_i \, \text{factor}_{i,t} + \beta_{n+1} \, \text{index}_t
 \]

- **Trading Signal:**
 If \(\hat{r}_{t+1} > 0\), then buy the index; otherwise, do not buy.
- Evaluate the strategy by observing the cumulative return in the period 2025/01/02 to 2025/03/01 and check how the inclusion of the sentiment index impacts the model’s \( R^2 \).

---

## Project Workflow

1. **Data Acquisition:**
 Gather labeled sentiment datasets (Positive, Neutral, Negative) for financial textual data from Github, Kaggle, and Hugging Face.

2. **BERT Model Fine-Tuning:**
 Fine-tune a Chinese BERT model enhanced with a fully connected layer to output sentiment probabilities.

3. **Web Scraping:**
 Utilize Beautiful Soup to scrape financial forum posts and comments from EastMoney-Shanghai Securities Composite index (SSEC) forum吧 within the target date range.

4. **Sentiment Calculation:**
 For each piece of scraped text, compute a sentiment score using the fine-tuned BERT model.

5. **Rolling Window Sentiment Index:**
 Aggregate sentiment scores over rolling monthly windows to compute the sentiment index using the formulas provided above.

6. **Predictive Regression & Strategy Design:**
 Integrate the sentiment index into a regression model and design a trading strategy based on predicted returns.

7. **Backtesting:**
 Evaluate the trading strategy’s performance during a specified period by comparing cumulative returns and looking at changes in the regression model performance.

---

## Project Structure

```
Stock-Market-Sentiment-Analysis/
├── data/
│ ├── raw/ # Raw sentiment data from Github, Kaggle, Hugging Face
│ ├── scraped/ # Scraped posts and comments from EastMoney-Shanghai Securities Composite index (SSEC) forum吧
│ └── processed/ # Preprocessed datasets for model training & analysis
├── models/
│ ├── bert_sentiment/# Fine-tuned BERT model for sentiment scoring
│ └── regression_model/# Regression models incorporating financial factors and sentiment index
├── notebooks/
│ ├── data_exploration.ipynb # Exploratory data analysis of collected datasets
│ ├── sentiment_analysis.ipynb # BERT fine-tuning and testing workflow
│ └── backtesting.ipynb# Evaluation of the trading strategy and backtesting results
├── scripts/
│ ├── web_scraping.py# Script to crawl and extract data using Beautiful Soup
│ └── calculate_index.py # Script to compute the monthly sentiment index from scraped data
├── README.md# This file
└── requirements.txt # List of project dependencies
```

---

## Installation

1. **Clone the Repository:**

 ```bash
 git clone https://github.com/your_username/Stock-Market-Sentiment-Analysis.git
 cd Stock-Market-Sentiment-Analysis
 ```

2. **Setup Virtual Environment and Install Dependencies:**

 ```bash
 python -m venv venv
 source venv/bin/activate# On Windows: venv\Scripts\activate
 pip install -r requirements.txt
 ```

3. **Download Additional Resources:**

- Ensure access to the necessary Chinese financial sentiment datasets.
- Download the required pre-trained Chinese BERT model.

---

## Usage

- **Data Preprocessing & Web Scraping:**

Run the following scripts to scrape data and compute the sentiment index:

```bash
python scripts/web_scraping.py
python scripts/calculate_index.py
```

- **Model Training & Evaluation:**

Navigate to the `notebooks` folder. Open and run:

- `sentiment_analysis.ipynb` to fine-tune and evaluate the BERT model.
- `backtesting.ipynb` for regression modeling, strategy design, and performance backtesting.

---

## Expected Results & Analysis

- **Sentiment Scoring:**
The fine-tuned BERT model outputs sentiment probabilities for each financial post/comment.

- **Monthly Sentiment Index:**
Aggregated using a rolling window method:

\[
\text{index}_t = \ln\left(\frac{1+\text{pos}_t}{1+\text{neg}_t}\right)
\]

- **Predictive Regression:**
The regression model:

\[
r_{t+1} = \alpha + \sum_{i=1}^{n} \beta_i\, \text{factor}_{i,t} + \beta_{n+1}\, \text{index}_t + \epsilon_t
\]

The inclusion of \(\text{index}_t\) is tested by its impact on the \( R^2 \) value and the significance of \(\beta_{n+1}\).

- **Trading Strategy:**
A simple rule-based strategy is implemented where, if:

\[
\hat{r}_{t+1} = \alpha + \sum_{i=1}^{n} \beta_i \, \text{factor}_{i,t} + \beta_{n+1} \, \text{index}_t > 0,
\]

then buy the index; otherwise, do not buy. Backtesting over the period 2025/01/02 to 2025/03/01 provides insights into the cumulative returns.

---

## Discussion & Future Work

- **Comparison with Existing Solutions:**
- Traditional sentiment analysis often relies on lexicon-based methods or delayed market signals (e.g., likes/upvotes).
- Our method utilizes real-time textual sentiment from financial forums, avoiding biases introduced by future engagement metrics.

- **Improvements & New Perspectives:**
- Experiment with alternative weighting schemes or credibility scores to enhance sentiment aggregation.
- Integrate additional financial indicators and explore advanced deep learning architectures.
- Compare our approach with other state-of-the-art methods to further validate performance.

---

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit an issue or pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or further information, please reach out to [your.email@example.com](mailto:your.email@example.com).

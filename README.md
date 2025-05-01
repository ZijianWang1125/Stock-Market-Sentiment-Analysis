# Stock Market Sentiment Analysis

Investor sentiment plays a crucial role in influencing stock market movements, but capturing and quantifying real-time sentiment remains challenging. In this project, we developed a sentiment analysis pipeline by crawling 71,888 posts from Eastmoney’s Shanghai Composite Index forum. We fine-tuned a Chinese ELECTRA model to classify post sentiment into three categories and constructed a rolling sentiment index to track market emotions dynamically.

## Project Structure

- `/data/electra_sentiment_chinese/`: Contains the Chinese sentiment analysis model and data

## Data Source

- `train_data_1`: https://tianchi.aliyun.com/dataset/158814
- `train_data_2`: https://tianchi.aliyun.com/dataset/179229
- `train_data_3`: https://github.com/algosenses/Stock_Market_Sentiment_Analysis/blob/master/data/positive.txt
- `train_data_4`: https://github.com/algosenses/Stock_Market_Sentiment_Analysis/blob/master/data/negative.txt
- `test_data`: Scraped by `../script/data_test_scraper.py`.


where

- `train_data_1`: `./data/electra_sentiment_chinese/train_data/train_data_1.xlsx`
- `train_data_2`: `./data/electra_sentiment_chinese/train_data/train_data_2.csv`
- `train_data_3`: `..data/electra_sentiment_chinese/train_data/train_data_3.txt`
- `train_data_4`: `./data/electra_sentiment_chinese/train_data/train_data_4.txt`
- `test_data`: `./data/electra_sentiment_chinese/test_data/test_data.csv`

## Enviroment

See `\enviroment\requirement.txt`

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Stock-Market-Sentiment-Analysis1.git
cd Stock-Market-Sentiment-Analysis1
```

### 2. Set up the environment and activate

- **venv**:

```bash
python -m venv <venv>

source <venv>/bin/activate
```

- **conda**:

```bash
conda create <venv>

conda activate <venv>
```

### 3. Install dependencies

- **pip**:

```bash
pip install -r ./enviroment/requirements.txt
```

- **conda**:

```bash
conda install -r ./enviroment/requirements.txt
```

## Usage

You can either repeat the training pipeline (to predict the sentiment using the texts from `test_data`) or predict the sentiment of a Chinese sentence input.

### 1. Train a model

To train a model by yourself, simply execute the [`main.py`]().

### 2. Predict a sentence

To input a Chinese sentence about financial or economical topics, and predict the sentiment of it by the model:

- 1. **Unzip Model**: Download the [electra_sentiment_chinese.rar](https://drive.google.com/file/d/1tbCgXhmZKg1YwxcStiXLL7p62cwXZ9P2/view?usp=drive_link) and unzip it under `./model/electra_sentiment_chinese`
- 2. **Unzip Data**: Download the [data.rar](https://drive.google.com/file/d/1Ahe7L4D7Dd959I7F31zxbFSVc-oxwDdq/view?usp=drive_link) and unzip it under `./model/electra_sentiment_chinese`
- 2. Second, execute ['model_finetuner.predict_sentiment']().

## Results

The sentiment analysis model classifies text into three categories:
- Positive: Indicating optimistic market sentiment
- Neutral: Indicating balanced or uncertain market views
- Negative: Indicating pessimistic market sentiment

Example analysis of recent market news:

| Date       | Source         | Text                           | Sentiment | Confidence |
| ---------- | -------------- | ------------------------------ | --------- | ---------- |
| 2023-04-15 | Financial News | 经济数据好于预期，市场上涨     | Positive  | 0.89       |
| 2023-04-16 | Social Media   | 投资者对新政策持观望态度       | Neutral   | 0.75       |
| 2023-04-17 | Market Forum   | 通胀数据令人担忧，可能引发抛售 | Negative  | 0.82       |

## License

MIT License

Copyright (c) 2023 Stock Market Sentiment Analysis Project

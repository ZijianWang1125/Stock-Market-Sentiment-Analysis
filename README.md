# Stock Market Sentiment Analysis

Investor sentiment plays a crucial role in influencing stock market movements, but capturing and quantifying real-time sentiment remains challenging. In this project, we developed a sentiment analysis pipeline by crawling 71,888 posts from Eastmoney’s Shanghai Composite Index forum. We fine-tuned a Chinese ELECTRA model to classify post sentiment into three categories and constructed a rolling sentiment index to track market emotions dynamically.

## Project Structure

Key files based on different [Usage](#usage) have sturcture like:

```
.
├── data/ # Data files for usage: "Train model to predict whole test data" only
│   └── electra_sentiment_chinese/
│       ├── train_data/           # Training datasets from various sources
│       │   ├── train_data_1.xlsx # Dataset from Alibaba Tianchi
│       │   ├── train_data_2.csv  # Dataset from Alibaba Tianchi
│       │   ├── train_data_3.txt  # Positive sentiment dataset from GitHub
│       │   ├── train_data_4.txt  # Negative sentiment dataset from GitHub
│       │   └── train_data_clean.txt # Processed training data
│       ├── test_data/            # Test datasets for model evaluation
│       │   ├── test_data.csv     # Raw scraped test data
│       │   └── test_data_clean.csv # Processed test data
│       └── pred_data/            # Output directory for prediction results
├── model/ # Fine-tuned Chinese ELECTRA model files for usage: "Use trained model to predict several input comments" only
│   └── electra_sentiment_chinese/
│       ├── config.json           # Model configuration
│       ├── model.safetensors     # Model weights
│       ├── special_tokens_map.json # Special tokens configuration
│       ├── tokenizer_config.json # Tokenizer configuration
│       └── vocab.txt             # Model vocabulary
├── script/
│   ├── data_test_loader.py       # Formatter to the test data: `./data/electra_sentiment_chinese/test_data`
│   ├── data_test_crawler.py       # Crawler to the test data: `./data/electra_sentiment_chinese/test_data`
│   ├── data_train_loader.py       # Formatter to the train data: `./data/electra_sentiment_chinese/train_data`
│   ├── main.py                   # Main script for training and evaluating the model
│   ├── model_finetuner.py        # Script for fine-tuning the ELECTRA model
│   └── model_predictor.py        # Script for predicting sentiment of input │
└── README.md                     # Project documentation
└── requirements.txt              # Dependencies for the project
```

## Data Source

After finishing the **Unzip** step in [Train model to predict whole test data](#train-a-model) part, you will be able to find several datasets in `./data/train_data` or `./data/test_data`. Here are their sources:

- `train_data_1`: https://tianchi.aliyun.com/dataset/158814
- `train_data_2`: https://tianchi.aliyun.com/dataset/179229
- `train_data_3`: https://github.com/algosenses/Stock_Market_Sentiment_Analysis/blob/master/data/positive.txt
- `train_data_4`: https://github.com/algosenses/Stock_Market_Sentiment_Analysis/blob/master/data/negative.txt
- `test_data`: Scraped by `../script/data_test_scraper.py`.

## Enviroment

See `requirements.txt` for a list of necessary Python packages.


## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Stock-Market-Sentiment-Analysis1.git
cd Stock-Market-Sentiment-Analysis1
```

### 2. Set up the environment and activate

Choose either `venv` or `conda`:

*   **venv**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

*   **conda**:
    ```bash
    conda create --name risk_env python=3.9 # Or desired Python version
    conda activate risk_env
    ```

### 3. Install dependencies

```bash
pip install -r ./requirements.txt
```

## Usage

You can either repeat the training pipeline to predict the sentiment using the comments from `./data/test_data` or use a trained model to predict the sentiment of several input Chinese comments.

### Train model to predict whole test data

To train model to predict whole test data by yourself:

1. **Download**: Download the `data.rar` in [Google Drive](https://drive.google.com/file/d/1OKVXTAq2P8ucE6wDyY_a4c9nYJSGOSuD/view?usp=sharing).

2. **Unzip**: Unzip the `data.rar` in root directory `./`.

3. **Execute**: Execute [`main.py`](https://github.com/ZijianWang1125/Stock-Market-Sentiment-Analysis/blob/main/script/main.py) and you will be given a model trained by yourself in `./model/electra_sentiment_chinese` and a predicted dataset in `./data/electra_sentiment_chinese/pred_data` based on a testing dataset in `./data/electra_sentiment_chinese/test_data/test_data_clean.csv`.

### Use trained model to predict several input comments

To use trained model to predict several input comments:

1. **Download**: Download the `model.rar` in [Google Drive](https://drive.google.com/file/d/1_zjliGZdjHmbQFrZnORcw9wE1TfqEWTc/view?usp=sharing).

2. **Unzip**: Unzip the `model.rar` in root directory `./`.

3. **Execute**: Execute [`model_predictor`](https://github.com/ZijianWang1125/Stock-Market-Sentiment-Analysis/blob/main/script/model_finetuner.py) and you will be required to input Chinese comments about stock topics in your desired quantity. After inputting, you will be given a sentiment analysis for these comments.

## Results

The sentiment analysis model classifies comment into three categories:

- Positive: Indicating optimistic market sentiment, with sentiment value of **2**
- Neutral: Indicating balanced or uncertain market views, with sentiment value of **1**
- Negative: Indicating pessimistic market sentiment with, sentiment value of **0**

Here are different results example for different usages:

- Example sentiment analysis of market comments in [Train model to predict whole test data](#train-a-model-to-predict-whole-test-data) part:

  | User Age | Comment # | Influecen Power | Read # |           Comment           |    Data    | Sentiment Value | Confidence | Weighted Sentiment Value |
  | :------: | :-------: | :-------------: | :----: | :-------------------------: | :--------: | :-------------: | :--------: | :----------------------: |
  |   8.59   |     0     |      1.93       |   0    |         国内科创板          | 2019-03-28 |        2        |    0.56    |           1.50           |
  |   0.04   |     0     |       0.5       |  200   |       科创板融资问题        | 2019-03-28 |        2        |    0.65    |           1.58           |
  |   0.25   |     0     |       1.0       |  574   | 缩量都能把股指拉红神勇的大A | 2019-03-27 |        2        |    0.73    |           1.59           |

- Example sentiment analysis of market comments in [Use trained model to predict several input comments](#use-trained-model-to-predict-several-comments) part:

  |            Comment             | Sentiment | Confidence | Weighted Sentiment |
  | :----------------------------: | :-------: | :--------: | :----------------: |
  |   经济数据好于预期，市场上涨   | Positive  |    0.89    |        1.88        |
  |    投资者对新政策持观望态度    |  Neutral  |    0.75    |        1.31        |
  | 通胀数据令人担忧，可能引发抛售 | Negative  |    0.82    |        0.03        |

## License

MIT License

Copyright (c) 2025 Stock Market Sentiment Analysis Project

# %%
import pandas as pd
import torch
from data_train_loader import load_train_data
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from model_finetuner import finetune_chinese_electra_for_sentiment
from model_predictor import predict_sentiment
from data_test_loader import load_test_data
from data_test_crawler import crawl_test_data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv_clean_path = r"../data/electra_sentiment_chinese/train_data/train_data_clean.csv"
    try:
        train_df_clean = pd.read_csv(train_csv_clean_path)
        print(f"Training data already exists at {train_csv_clean_path}.")
    except:
        train_paths = (
            rf"../data/electra_sentiment_chinese/train_data/train_data_{suffix}"
            for suffix in ("1.xlsx", "2.csv", "3.txt", "4.txt")
        )
        train_df_clean = load_train_data(train_paths)
        train_df_clean.to_csv(train_csv_clean_path, index=False)
        print(f"Training data saved to {train_csv_clean_path}")

    models_path = r"../model/electra_sentiment_chinese"
    try:
        tokenizer = ElectraTokenizer.from_pretrained(models_path)
        model = ElectraForSequenceClassification.from_pretrained(
        models_path,
        num_labels=3,
        ignore_mismatched_sizes=True
        ).to(device)
        print(f"Model and tokenizer loaded from {models_path}")
    except:
        model, tokenizer = finetune_chinese_electra_for_sentiment(
            train_df_clean,
            models_path
        )
        print(f"Model and tokenizer saved to {models_path}")

    test_csv_clean_path = r"../data/electra_sentiment_chinese/test_data/test_data_clean.csv"
    try:
        test_df_clean = pd.read_csv(test_csv_clean_path)
        print(f"Test data already exists at {test_csv_clean_path}.")
    except:
        test_csv_path = r"../data/electra_sentiment_chinese/test_data/test_data.csv"
        try:
            test_df_clean = load_test_data(test_csv_path)
            test_df_clean.to_csv(test_csv_clean_path, index=False)
            print(f"Test data cleaned and saved to {test_csv_clean_path}")
        except:
            test_df = crawl_test_data()
            test_df.to_csv(test_csv_clean_path, index=False)
            test_df_clean = load_test_data(test_csv_path)
            test_df_clean.to_csv(test_csv_clean_path, index=False)
            print(f"Test data saved to {test_csv_clean_path}")

    pred_csv_path = r"../data/electra_sentiment_chinese/pred_data/pred_data.csv"
    try:
        pred_df = pd.read_csv(pred_csv_path)
        print(f"Prediction data already exists at {pred_csv_path}.")
    except:
        # Predict sentiment
        pred_df = predict_sentiment(model, tokenizer, test_df_clean)
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Prediction data saved to {pred_csv_path}")

# %%
if __name__ == "__main__":
    main()

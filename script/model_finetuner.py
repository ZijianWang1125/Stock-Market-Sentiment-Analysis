# %%
import pandas as pd
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_train_loader import load_train_data

random_state = 42

# Create dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, device=None):
        self.encodings = encodings
        self.labels = labels
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 3. Fine-tuning function
def finetune_chinese_electra_for_sentiment(
    train_df,
    output_dir
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Split into training and validation sets
    train_train_df, eval_train_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=random_state,
        stratify=train_df['label']
    )
    
    # Load Chinese ELECTRA model and tokenizer
    model_name = "hfl/chinese-electra-base-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Tokenize the data
    train_encodings = tokenizer(
        train_train_df['title'].tolist(),
        truncation=True,
        max_length=128,
        padding=True
    )
    eval_encodings = tokenizer(
        eval_train_df['title'].tolist(),
        truncation=True,
        max_length=128,
        padding=True
    )
    
    # Create datasets
    train_dataset = SentimentDataset(
        train_encodings,
        train_train_df['label'].tolist(),
        device=device
    )
    eval_dataset = SentimentDataset(
        eval_encodings,
        eval_train_df['label'].tolist(),
        device=device
    )
    
    # Define evaluation function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='weighted'
        )

        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # 减少轮数，因为数据量大
        per_device_train_batch_size=32,  # 增加批量大小，加快训练
        per_device_eval_batch_size=128,  # 增加评估批量
        warmup_ratio=0.05,  # 减小预热比例
        weight_decay=0.01,  # 保持正则化
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,  # 增加日志步数
        eval_strategy="steps",
        eval_steps=1000,  # 增加评估间隔
        save_strategy="steps",
        save_steps=1000,  # 与评估间隔保持一致
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,  # 限制保存的检查点数量
        fp16=True,  # 保持半精度训练
        gradient_accumulation_steps=8,  # 增加梯度累积步数
        learning_rate=3e-5,  # 略微提高学习率
        lr_scheduler_type="linear",  # 对大数据集使用线性调度器
        push_to_hub=False,
        save_safetensors=True,
        greater_is_better=True,
        no_cuda=False,
        dataloader_num_workers=8,  # 增加数据加载线程
        group_by_length=True,  # 保持按长度分组
        max_grad_norm=1.0,  # 添加梯度裁剪
        gradient_checkpointing=True,  # 启用梯度检查点以节省内存
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        #callbacks=[
        #    EarlyStoppingCallback(early_stopping_patience=2)
        #],  # Stop if no improvement after 2 rounds
    )

    # Fine-tune the model
    print("Starting model training...")
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return model, tokenizer

# 4. Using the fine-tuned model for prediction
def predict_sentiment(model, tokenizer, test_df):
    device = next(model.parameters()).device
    
    # Create result lists
    emotions = []
    
    # Process each title
    for title in test_df['title']:
        inputs = tokenizer(
            title,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        # Move inputs to the model's device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        emotion = torch.argmax(logits, dim=1).item()
        
        emotions.append(emotion)
    
    # Create results DataFrame
    pred_df = test_df
    pred_df['emotion'] = emotions
    
    return pred_df


# %%
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_csv_clean_path = r"../data/electra_sentiment_chinese/train_data/train_data_clean.csv"

    try:
        train_df_clean = pd.read_csv(train_csv_clean_path)
    except:
        train_paths = (
        rf"../data/electra_sentiment_chinese/train_data/train_data_{suffix}"
        for suffix in ("1.xlsx", "2.csv", "3.txt", "4.txt")
    )
        train_df_clean = load_train_data(train_paths)
        train_df_clean.to_csv(train_csv_clean_path, index=False)

    models_path = r"../model/electra_sentiment_chinese"
    try:
        tokenizer = ElectraTokenizer.from_pretrained(models_path)
        model = ElectraForSequenceClassification.from_pretrained(
        models_path,
        num_labels=3,
        ignore_mismatched_sizes=True
        ).to(device)
    except:
        model, tokenizer = finetune_chinese_electra_for_sentiment(
            train_df_clean,
            models_path
        )

    # Test the fine-tuned model
    test_titles = [
        "这部电影真的很棒，我非常喜欢！",
        "服务态度一般，不算好也不算差。",
        "产品质量太差了，完全不值这个价格。"
    ]

    # Create a DataFrame with test titles
    test_df = pd.DataFrame({'title': test_titles})
    
    # Predict sentiment
    pred_df = predict_sentiment(model, tokenizer, test_df)

    print("\nPrediction results:")
    for i, row in pred_df.iterrows():
        emotion_label = "negative" if row['emotion'] == -1 else "neutral" if row['emotion'] == 0 else "positive"
        print(f"Title: {row['title']}")
        print(f"Emotion: {emotion_label} (Sentiment value: {row['emotion']})")
        print("")

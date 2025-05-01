import pandas as pd
import numpy as np
import torch

# Using the fine-tuned model for prediction
def predict_sentiment(model, tokenizer, test_df, batch_size=32):
    device = next(model.parameters()).device
    
    # 存储所有批次的结果
    all_emotions = []
    all_confidences = []
    all_weighted_emotions = []
    
    # 分批处理
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        
        # 为当前批次处理标题
        inputs = tokenizer(
            batch_df['title'].tolist(),
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # 将输入转移到模型所在的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        
        # 获取每个输入的预测情感类别
        emotions = torch.argmax(logits, dim=1).cpu()
        
        # 应用softmax获取概率
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # 获取每个预测的置信度
        confidences = probs[torch.arange(probs.size(0)), emotions].cpu().numpy()
        
        # 创建情感索引张量 [0, 1, 2] 
        emotion_indices = torch.arange(probs.size(1), device=probs.device, dtype=torch.float)

        # 使用矩阵乘法计算加权情感
        weighted_emotions = torch.matmul(probs, emotion_indices).cpu().numpy()
        
        # 将批次结果添加到总列表
        all_emotions.append(emotions)
        all_confidences.append(confidences)
        all_weighted_emotions.append(weighted_emotions)
        
        # 释放GPU内存
        del inputs, outputs, logits, probs, emotions
        torch.cuda.empty_cache()
        
        # 输出进度
        print(f"Processed batch {i//batch_size + 1}/{(len(test_df) + batch_size - 1)//batch_size}", end="\r")
    
    # 合并所有批次的结果
    all_emotions = torch.cat(all_emotions).numpy()
    all_confidences = np.concatenate(all_confidences)
    all_weighted_emotions = np.concatenate(all_weighted_emotions)
    
    # 创建结果DataFrame
    pred_df = test_df.copy()  # 使用copy避免修改原始数据框
    pred_df['emotion'] = all_emotions
    pred_df['confidence'] = all_confidences
    pred_df['weighted_emotions'] = all_weighted_emotions
    
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

    n_titles = int(input("Enter the number of sentences to predict sentiment: "))
    if n_titles <= 0:
        print("Invalid number of sentences. Exiting.")
        exit()

    # Test the fine-tuned model
    test_titles = [
        input(f"Enter sentence #{i} to predict sentiment: ") for i in range(n_titles)
    ]

    # Create a DataFrame with test titles
    test_df = pd.DataFrame({'title': test_titles})

    # Predict sentiment
    pred_df = predict_sentiment(model, tokenizer, test_df)

    print("\nSentiment Analysis:")
    for i, row in pred_df.iterrows():
        title = row['title']
        emotion = row['emotion']
        emotion_label = "negative" if emotion == 0 else "neutral" if emotion == 1 else "positive"
        confidence = row['confidence']
        weighted_emotions = row['weighted_emotions']
        weighted_emotions_label = "negative" if weighted_emotions < 0.5 else "neutral" if weighted_emotions == 0.5 else "positive"

        print(f"Sentence: {title}")
        print(f"Sentiment: {emotion_label} (Confidence: {confidence:.4f})")
        print(f"Weighted Sentiment: {weighted_emotions_label}\n")
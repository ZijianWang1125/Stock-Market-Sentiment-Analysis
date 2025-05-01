# %%
import pandas as pd

def load_train_data(train_paths):
    train_dfs = []

    # 将txt文件转换为csv文件
    def convert_txt_to_csv(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Split the lines into title and label
        data = [line.replace(" ", "") for line in lines]
        train_df = pd.DataFrame(data, columns=['title'])

        return train_df

    for path in train_paths:
        if path.endswith('1.xlsx'):
            train_df = pd.read_excel(path)
            # Ensure data columns exist
            
            # 删除无关列
            train_df = train_df[['title', 'emotion']]

            # Remove rows with missing values
            train_df.dropna(subset=['title', 'emotion'], inplace=True)

            # Map emotion values from {-1, 0, 1} to {0, 1, 2}
            # where: -1 -> 0 (negative), 0 -> 1 (neutral), 1 -> 2 (positive)
            train_df['emotion'] = train_df['emotion'].astype('int8')
            train_df['label'] = train_df['emotion'] + 1
            train_df.drop(columns=['emotion'], inplace=True)
            train_df = train_df[train_df['label'].isin([0, 1, 2])]
            
            train_dfs.append(train_df)

        elif path.endswith('2.csv'):
            train_df = pd.read_csv(path)

            train_df = train_df[['详情标题', 'sentiment']]
            train_df.rename(
                columns={
                    '详情标题': 'title',
                    'sentiment': 'label'
                },
                inplace=True
            )

            # Remove rows with missing values
            train_df.dropna(subset=['title', 'label'], inplace=True)

            train_df['label'] = (train_df['label'] * 2).round()
            train_df['label'] = train_df['label'].astype('int8')
            train_df = train_df[train_df['label'].isin([0, 1, 2])]

            train_dfs.append(train_df)

        elif path.endswith('.txt'):
            # Convert txt to csv
            train_df = convert_txt_to_csv(path)
            
            # Remove rows with missing values
            train_df.dropna(subset=['title'], inplace=True)

            label_value = 2 if path.endswith('3.txt') else 0
            train_df['label'] = label_value
            train_df['label'] = train_df['label'].astype('int8')

            train_dfs.append(train_df)
            
    train_df_clean = pd.concat(train_dfs, ignore_index=True)
    return train_df_clean


# %%
# Example usage
train_paths = (
    rf"../data/electra_sentiment_chinese/train_data/train_data_{suffix}"
    for suffix in ("1.xlsx", "2.csv", "3.txt", "4.txt")
)
train_csv_clean_path = r"../data/electra_sentiment_chinese/train_data/train_data_clean.csv"

# Load the training data
train_df_clean = load_train_data(train_paths)
train_df_clean.to_csv(train_csv_clean_path, index=False)

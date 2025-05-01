# %%
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def load_test_data(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    # Ensure data columns exist

    # 删除无关列
    test_df.drop(columns=['author'], inplace=True)

    test_df.rename(
        columns={
            'pinglun_n': 'comment_n'
    }, inplace=True)    

    # Step 2: Extract and format 'date' from 'post_time'
    test_df['date'] = test_df['post_time'].str.split(',').str[0]
    test_df['date'] = pd.to_datetime(test_df['date'], format='%Y-%m-%d', errors='coerce')
    test_df['date'] = test_df['date'].dt.strftime('%Y-%m-%d')

    # Step 3: Drop the original 'post_time' column
    test_df.drop(columns=['post_time'], inplace=True)

    # Step 4: Fill missing values in 'comment_n' and 'read_n' with 0
    test_df['comment_n'] = test_df['comment_n'].fillna(0)
    test_df['read_n'] = test_df['read_n'].fillna(0)

    # Step 5: Drop the 'text' column due to excessive missing values
    test_df = test_df.drop(columns=['text'])

    # Step 7: Apply IterativeImputer to impute missing values in numerical columns
    num_cols = ['age', 'power', 'comment_n', 'read_n']
    test_df_numeric = test_df[num_cols]

    # Initialize IterativeImputer
    iterative_imputer = IterativeImputer(
        estimator=BayesianRidge(),  # Use BayesianRidge as the estimator
        max_iter=10,
        random_state=42
    )

    # Fit and transform the numeric data
    array_imputed = iterative_imputer.fit_transform(test_df_numeric)

    # Replace numeric columns with imputed values
    test_df_imputed = pd.DataFrame(array_imputed, columns=num_cols)
    test_df[num_cols] = test_df_imputed

    # Remove rows with missing values
    test_df.dropna(inplace=True)

    test_df['comment_n'] = test_df['comment_n'].astype('int16')
    test_df['read_n'] = test_df['read_n'].astype('int16')
    test_df_clean = test_df

    return test_df_clean

# %%
if __name__ == "__main__":
    test_csv_path = r"../data/electra_sentiment_chinese/test_data/test_data.csv"
    test_csv_clean_path = r"../data/electra_sentiment_chinese/test_data/test_data_clean.csv"

    test_df_clean = load_test_data(test_csv_path)
    test_df_clean.to_csv(test_csv_clean_path, index=False)

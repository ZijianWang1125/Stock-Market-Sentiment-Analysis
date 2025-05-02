# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def crawl_page(url):
    """Crawl a single page and extract post data"""
    data = []
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = soup.find_all('div', class_='articleh')

        for post in posts:
            try:
                # Author
                author_tag = post.find('span', class_='l4 a4')
                author = author_tag.get_text(strip=True) if author_tag else None

                # Age and Power
                age = None
                power = None
                user_info = post.find('span', class_='l5 a5')
                if user_info and user_info.get('title'):
                    info = user_info['title'].split()
                    for item in info:
                        if '吧龄' in item:
                            try:
                                age = float(item.replace('吧龄', '').replace('年', ''))
                            except:
                                age = None
                        if '影响力' in item:
                            try:
                                power = float(item.replace('影响力', '').replace('万', ''))
                                power = power * 10000 if '万' in item else power
                            except:
                                power = None

                # Title
                title_tag = post.find('a', class_='note')
                title = title_tag.get_text(strip=True) if title_tag else None

                # Read and Comment Numbers
                read_tag = post.find('span', class_='l1 a1')
                read_n = float(read_tag.get_text(strip=True).replace(',', '')) if read_tag else 0

                comment_tag = post.find('span', class_='l2 a2')
                pinglun_n = float(comment_tag.get_text(strip=True).replace(',', '')) if comment_tag else 0

                # Date
                date_tag = post.find('span', class_='l6 a6')
                date = date_tag.get_text(strip=True) if date_tag else None

                # Standardize date format
                try:
                    date = pd.to_datetime(date, errors='coerce')
                    date = date.strftime('%Y-%m-%d') if pd.notnull(date) else None
                except:
                    date = None

                # Append
                data.append({
                    'author': author,
                    'age': age,
                    'power': power,
                    'title': title,
                    'pinglun_n': pinglun_n,
                    'read_n': read_n,
                    'date': date
                })

            except Exception as e:
                print(f"Error parsing a post: {e}")
                continue

    except Exception as e:
        print(f"Error fetching page: {e}")
    
    return data

def save_data(data, output_path):
    """Save data to CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df

def crawl_test_data(
    start_page=1,
    end_page=100,
    save_interval=1000, 
    output_path='data/data.csv',
    base_url='https://guba.eastmoney.com/list,sz000001,f_{}.html'
):
    """Main function to crawl Eastmoney Guba"""
    data = []
    post_counter = 0

    for page in range(start_page, end_page + 1):
        url = base_url.format(page)
        print(f'Crawling page {page}...')
        
        page_data = crawl_page(url)
        data.extend(page_data)
        post_counter += len(page_data)
        
        # Save periodically
        if post_counter > 0 and post_counter % save_interval == 0:
            save_data(data, output_path)
            print(f'Saved {post_counter} posts so far...')
            
        # Random sleep to avoid being banned
        time.sleep(random.uniform(1, 1.5))
    
    # Final save
    final_df = save_data(data, output_path)
    print(f'Done! Total {post_counter} posts saved to {output_path}')
    
    return final_df

# %%
if __name__ == '__main__':
    
    # Call the main function
    df = crawl_test_data()
    df.to_csv(r"../data/electra_sentiment_chinese/test_data/test_data.csv", index=False, encoding='utf-8-sig')

# %%

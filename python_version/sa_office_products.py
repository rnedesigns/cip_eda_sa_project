"""
These are the functions written to perform EDA, Cleaning dataset and Sentiment Analysis on office_products category.
"""

import json
import nltk.sentiment

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

analyzer = nltk.sentiment.SentimentIntensityAnalyzer()

"""
Data Cleaning the raw dataset by removing unwanted variables using del_keys() function:
"""


def del_key():
    with open('../datasets_raw/reviews_ori_data.json') as data_file:
        products_data = json.load(data_file)

        print(len(products_data['musical_instruments']))
        print(len(products_data['software_products']))
        print(len(products_data['office_products']))

    for item_ in products_data['office_products']:

        for key_ in ['verified', 'reviewerName', 'reviewTime', 'unixReviewTime']:
            item_.pop(key_)

    with open('../datasets_output/office_products_dir/office_products_cleaned.json', 'w') as data_file:
        json.dump(products_data['office_products'], data_file, indent=4)


del_key()

"""
Exploratory Data Analysis
"""
# To view contents of cleaned dataset from del_keys() step:
office_products_cleaned_ = pd.read_json('../datasets_output/office_products_dir/office_products_cleaned.json')
print(office_products_cleaned_)

# Data types of each variables/columns:
print(office_products_cleaned_.dtypes)

# Descriptive summary of cleaned dataset:
print(office_products_cleaned_.describe())

# Number of rows and columns in cleaned dataset:
print(office_products_cleaned_.shape)

# General information about cleaned dataset:
print(office_products_cleaned_.info())

"""
Performing Sentiment Analysis
"""
users_review_text = []
products_id = []
sentiment_scores = []

with open('../datasets_output/office_products_dir/office_products_cleaned.json') as file:
    data = json.load(file)

    for item in data:
        # review text
        review_text = item['reviewText']
        # print(review_text.replace("\n\n", " ").replace("  ", " "))
        users_review_text.append(review_text.replace("\n\n", " ").replace("  ", " "))

        product_id = item['asin']
        products_id.append(product_id)

        # sentiment score
        scores = analyzer.polarity_scores(review_text)
        sentiment_score = scores['compound']

        sentiment_scores.append(sentiment_score)

# To create bar graph for each product category:
bar_graph_data = {
    'x': products_id,
    'y': sentiment_scores
}

plt.figure(figsize=(9, 12))
sns.barplot(data=bar_graph_data, x='x', y='y')
plt.title("Sentiment scores of each product in Office-Products category")
plt.xlabel("Product IDs")
plt.ylabel("Sentiment Scores")
plt.xticks(rotation=90)
plt.savefig('../datasets_output/office_products_dir/office_products_senti_graph.png')

# Create Pandas DataFrame:
data = {
    'user_review': users_review_text,
    'product_id': products_id,
    'sentiment_score': sentiment_scores
}

reviews_df = pd.DataFrame(data)
print(reviews_df)

reviews_df.to_csv('../datasets_output/office_products_dir/office_products_sa_df.csv', index=None)

print(reviews_df[['product_id']].value_counts())
print("Number of products in Office-Products category:", len(reviews_df[['product_id']].value_counts()))

# To compute the average (mean) of each product's sentiment score:
sentiment_avg_score = reviews_df.groupby('product_id').mean()[['sentiment_score']]  # returns df object

print(type(sentiment_avg_score))
print(sentiment_avg_score)
print(sentiment_avg_score.round({'sentiment_score': 2}) * 100)

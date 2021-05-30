"""
Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32

-> In this code, loading cleaned dataset generated from 'to_delete_key()' function.
-> Then cleaned each review text by removing empty and new line characters.
-> And computed sentiment score for each review and stored in the list object.
-> Plotted the bar graph to visually see the scores of each products in software_products category.
-> Next saved the dataset with sentiment scores of each products into amazon_products_sa_df.csv file.
-> Then computed average sentiment score for each product. (Note: each product can have multiple reviews)

Note: here I have not removed stop words as it contains 'not' and similar words in the stopwords which effects the
overall sentiment score of the such reviews containing 'not' and similar texts.
"""

import json
import nltk.sentiment

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

analyzer = nltk.sentiment.SentimentIntensityAnalyzer()

# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
#
# print(stopwords.words('english'))

users_review_text = []
products_id = []
sentiment_scores = []

with open('../datasets_output/software_products_cleaned.json') as file:
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
        # print("Sentiment score: ", sentiment_score)
        sentiment_scores.append(sentiment_score)

# To create bar graph for each product category:
bar_graph_data = {
    'x': products_id,
    'y': sentiment_scores
}

plt.figure(figsize=(9, 12))
sns.barplot(x='x', y='y', data=bar_graph_data)
plt.title("Sentiment scores of each product in Software-Products category")
plt.xlabel("Product IDs")
plt.ylabel("Sentiment Scores")
# plt.ylim(0, 5)
plt.xticks(rotation=90)
plt.savefig('../datasets_output/software_products_senti_graph.png')

# Create Pandas DataFrame:
data = {
    'user_review': users_review_text,
    'product_id': products_id,
    'sentiment_score': sentiment_scores
}

reviews_df = pd.DataFrame(data)
print(reviews_df)

reviews_df.to_csv('../datasets_output/software_products_sa_df.csv', index=None)

print(reviews_df[['product_id']].value_counts())
print("Number of products in Software-Products category:", len(reviews_df[['product_id']].value_counts()))

# To compute the average (mean) of each product's sentiment score:
sentiment_avg_score = reviews_df.groupby('product_id').mean()[['sentiment_score']]  # returns df object
# sentiment_avg_score = reviews_df.groupby('product_id').mean().sentiment_score  # returns series object
print(type(sentiment_avg_score))
print(sentiment_avg_score)
print(sentiment_avg_score.round({'sentiment_score': 2}) * 100)

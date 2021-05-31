### EDA and Sentiment Analysis - Amazon's Products Reviews datasets available from below URLs:
(Exploratory Data Analysis - EDA)
(Sentiment Analysis - SA)

The open-source datasets I have chosen are available at:
* https://snap.stanford.edu/data/web-Amazon.html
* https://nijianmo.github.io/amazon/index.html#code

The datasets I have chosen are: software & office products and musical instruments categories..

The code walkthrough for this, available @ https://youtu.be/cqbimjZjWH4

**Code brief:**
* delete_key(): to remove unwanted fields/variables from the original datasets.
* sentiment_analysis_amazon_reviews.py: in this file, I have written code to remove unwanted spaces and new line characters for the next step. From this cleaned json, bar graph and csv files are generated. CSV files are generated with the results containing user_reviews, product_id and sentiment score for each product.

Prior to SA, I have perfomed EDA to understand the datasets, which can be seen in 'cip_eda_sa_project/jupyter_nb_version/sa_software_products.ipynb' notebook.

**Description**: I have performed Sentiment Analysis on Amazon’s products review by their customers to help understand each product’s, category-wise customers satisfaction/experience, based on 'customers review' metric. The quality of analysis depends on the consistency of the metrics available for each product. So here I have considered the consistent & required features/variables/columns. SA, helps to perform quality check on products and sellers. Note that each product can have multiple reviews by the multiple customers!

**Analyses**: To meet KPIs, maintain the stability and growth of sales which directly proportional to the quality of the products and sellers of a business. More non-positive reactions indicates that, those product/category needs attention to improvise on its quality checks.

I have performed these common steps of SA for the selected products category and can be further refactored by writing generalised function to pass dataset as an argument to the caller function and have the re-usable functions.

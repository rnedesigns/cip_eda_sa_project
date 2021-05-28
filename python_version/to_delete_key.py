"""
Below function helps to delete unwanted keys from the dataset passed in and create new cleaned
json file with the given name:
"""

import json


def del_key():
    with open('datasets_raw/reviews_ori_data.json') as file:
        data = json.load(file)

        print(len(data['musical_instruments']))
        print(len(data['software_products']))
        print(len(data['office_products']))

    for item in data['software_products']:

        for key_ in ['verified', 'reviewerName', 'reviewTime', 'unixReviewTime']:
            item.pop(key_)

        software_products_cleaned = data['software_products']

    with open('datasets_output/software_products_cleaned.json', 'w') as file:
        json.dump(software_products_cleaned, file, indent=4)


del_key()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


def import_data(masterCategory):
    dropped_columns = ["usage", "filename"]
    df1 = pd.read_csv("data/styles.csv")
    df2 = pd.read_csv("data/images.csv")
    result = pd.concat([df1, df2], axis=1).drop(columns=dropped_columns)
    filtered_result = result[result["masterCategory"] == masterCategory]

    # Need to reset the index because when filtering, the original matrix index is kept
    filtered_result.reset_index(drop=True, inplace=True)
    return filtered_result


def compute_similarities_content(df, column):
    tfidf = TfidfVectorizer(stop_words="english")

    # Compare each cell to each unique word found in entire column as matrix
    tfidf_matrix = tfidf.fit_transform(df[column])

    # all the unique words found in the column
    # feature_names = tfidf.get_feature_names_out()
    # print(f"feature name len is: {len(feature_names)}")

    # Compute cosine similairty
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def get_item_details_from_index(index, df):
    return df.iloc[index]


def get_similar_products_content(df, search_column, description, similarity_matrix):
    matching_indexes = df.index[df[search_column].values == description]
    matching_values_df = df.loc[matching_indexes]
    print(f"df that contains matching index: \n{matching_values_df}")

    # Take the first index we find, possible to have multiple
    initial_index = int(matching_values_df.index[0])
    # initial_index = pd.Series(df.index)
    # print(f"initialindex is {initial_index}")
    # print(df.loc[initial_index])

    reverse_indices_map = pd.Series(df.index, index=df[search_column])
    print(f"REVERSEINDEX VALUE: {reverse_indices_map.index[initial_index]}")
    item_list = list(enumerate(similarity_matrix[initial_index]))

    # This filters the list except the initial index, simplify without filter first
    # similar_items = list(filter(lambda x:x[0] != int(initial_index), sorted(item_list,key=lambda x:x[1], reverse=True)))

    similar_items = sorted(item_list, key=lambda x: x[1], reverse=True)
    top_ten_similar_items = similar_items[:10]
    # print(f"THE TOP 10 SIMILAR ITEMS ARE \n {top_ten_similar_items}")

    for i, similarity_index in top_ten_similar_items:
        print(f"df index is {i}, similairty index is {similarity_index}")
        print(df[search_column].values[i])
        # print(get_item_details_from_index(i, df))


#     top_items = []
#     # print(top_ten_similar_items)
#     # print(len(top_ten_similar_items))
#     # print(df[6945:6955])
#     # for i in range(len(top_ten_similar_items)):
#     #     print(f"THIS IS I, {i}")
#     #     print(similar_items[i][0])
#     #     df.loc(similar_items[i][0])
#     #     # top_items.append(df.iloc[similar_items][i][0])
#     # print(f"Top items are \n {top_items}")

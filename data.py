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
    return filtered_result


def compute_similarities_content(df, column):
    tfidf = TfidfVectorizer(stop_words="english")

    # I'm searching for just the column productDisplayName
    print(f"this is my function \n {df[column]}")

    # Compare each cell to each unique word found in entire column as matrix
    # tfidf_matrix = tfidf.fit_transform(df[column].to_numpy().astype("U"))
    tfidf_matrix = tfidf.fit_transform(df[column])
    # print(f"this is my tfidf matrix \n{tfidf_matrix}")
    # print(tfidf_matrix.shape)

    # tfidf_array = tfidf_matrix.toarray()
    # print(tfidf_array)
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(len(tfidf_array[0]))

    # all the unique words found in the column
    feature_names = tfidf.get_feature_names_out()
    print(f"feature name len is: {len(feature_names)}")

    # Compute cosine similairty
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_sim)

    # cosine_sim_pd = pd.DataFrame(cosine_sim)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', 5)
    return cosine_sim


def get_item_details_from_index(index, df):
    return df.iloc[index]


def get_similar_products_content(df, search_column, description, similarity_matrix):
    # print(f"Dataframe:\n {df}")
    # print(df.shape)
    matching_indexes = df.index[df[search_column].values == description]
    print(f"matching_indexes are: {matching_indexes}")
    matching_values_series = df.loc[matching_indexes]
    print(f"series that contains matching index: \n{matching_values_series}")

    # # Take the first index we find, possible to have multiple
    initial_index = int(matching_values_series.index[0])
    # # print(f"initialindex is {initial_index}")
    # # print(df.loc[initial_index])
    # # print(type(initial_index))
    # # print(similarity_matrix)

    item_list = list(enumerate(similarity_matrix[initial_index]))
    # print(item_list)
    # print(item_list[9221])
    # print(sorted(item_list, reverse=True))

    # This filters the list except the initial index, simplify without filter first
    # similar_items = list(filter(lambda x:x[0] != int(initial_index), sorted(item_list,key=lambda x:x[1], reverse=True)))
    
    similar_items = sorted(item_list, key=lambda x: x[1], reverse=True)
    top_ten_similar_items = similar_items[:10]
    print(top_ten_similar_items)
    # print(df.loc[1942])
    # print(get_item_details_from_index(7712, df))

    for i, similarity_index in top_ten_similar_items:
        print(f"df index is {i}, similairty index is {similarity_index}")
        print(get_item_details_from_index(i, df))




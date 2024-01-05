from data import (
    import_data,
    compute_similarities_content,
    get_similar_products_content,
)


def main():
    masterCategory = "Footwear"
    data = import_data(masterCategory)

    # First pass, hardcode only one search column to find similarity
    search_column = "productDisplayName"
    similarity_matrix = compute_similarities_content(data, search_column)

    description = "Enroute Men Black Formal Shoes"
    get_similar_products_content(data, search_column, description, similarity_matrix)


if __name__ == "__main__":
    main()

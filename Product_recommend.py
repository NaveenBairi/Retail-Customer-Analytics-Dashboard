import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df=pd.read_csv("Cleaned_Retail_data")
print(df.head())
print(df.shape)
print(df.info())

df=df[['CustomerID','Description','Quantity']]
print(df.head())

df=df.dropna(subset=['CustomerID','Description'])
print(df.isnull().sum())

## creating the customer*product matrix
customer_product_matrix = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print(customer_product_matrix.shape)
print(customer_product_matrix.head())

product_similarity=cosine_similarity(customer_product_matrix.T)
product_df=pd.DataFrame(product_similarity,
                                 index=customer_product_matrix.columns,
                                 columns=customer_product_matrix.columns)
print(product_df.shape)
print(product_df.head())

# creating recommendation Function
def recommend_products(product_name, top_n=5):
    if product_name not in product_df.columns:
        return "product not found"
    similar_products=product_df[product_name]
    similar_products=similar_products.sort_values(ascending=False)
    recommendations = similar_products.iloc[1:top_n+1]
    return recommendations
print(df['Description'].unique()[:20])

print(recommend_products("WHITE HANGING HEART T-LIGHT HOLDER"))

result = recommend_products("WHITE HANGING HEART T-LIGHT HOLDER")
print(result)

# Save Product Similarity Matrix
# Save Product Similarity Matrix
joblib.dump(product_df, "product_similarity.pkl")
print("Product similarity matrix saved successfully!")

# Save Product List
product_list = df["Description"].drop_duplicates().sort_values()
joblib.dump(product_list, "product_list.pkl")
print("Product list saved successfully!")
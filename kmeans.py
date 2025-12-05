# ============================================================
# K-Means Clustering - PRICERUNNER DATASET
# Unsupervised Classification: Product Category Prediction
# ============================================================
# Problem Statement:
# - Goal: Cluster products into categories using machine learning
# - Input: Product Title (text)
# - Target: Category Label (10 categories)
# - Dataset: PriceRunner from UCI
# ============================================================


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def entropy(counts):
    total = sum(counts)
    ent = 0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        ent -= p * np.log2(p)
    return ent

def purity(counts):
    total = sum(counts)
    return max(counts) / total

#load dataset
df = pd.read_csv("pricerunner_aggregate.csv")
df = df.rename(columns={' Category Label': 'Category Label'})

#use only the product title as the feature
labels = df["Product Title"].astype(str)

#convert text labels into numeric vectors
#build vocab of max 200 dimensions
vectorizer = TfidfVectorizer(
    stop_words="english", 
    max_features=200,
    )
X = vectorizer.fit_transform(labels)

#compress into 10 dimensions
svd = TruncatedSVD(n_components=10, random_state=42)
X_reduced = svd.fit_transform(X)


#run K-means with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X_reduced)

#calculate silhouette score
score = silhouette_score(X_reduced, df["kmeans_cluster"])
print("Silhouette score:", score)

#calculate entropy and purity
categories = [
    "Mobile Phones",
    "TVs",
    "CPUs",
    "Digital Cameras",
    "Microwaves",
    "Dishwashers",
    "Washing Machines",
    "Freezers",
    "Fridge Freezers",
    "Fridges"
]

#get labels of true categories and computed clusters
true_labels = df["Category Label"].astype(str)
pred_labels = df["kmeans_cluster"]

cluster_ids = sorted(df["kmeans_cluster"].unique())

table_data = []

#count the occurrences of each category in the clusters
for cid in cluster_ids:
    rows_in_cluster = df[df["kmeans_cluster"] == cid]
    
    #category counts for this cluster
    counts = [sum(rows_in_cluster["Category Label"] == cat) for cat in categories]
    
    ent = entropy(counts)
    pur = purity(counts)
    
    row = [cid] + counts + [round(ent, 4), round(pur, 4)]
    table_data.append(row)

#calculate totals and overall entropy and purity
total_counts = [sum(true_labels == cat) for cat in categories]
overall_purity = sum(
    len(df[df["kmeans_cluster"] == cid]) * purity(
        [sum(df[df["kmeans_cluster"] == cid]["Category Label"] == cat) for cat in categories]
    )
    for cid in cluster_ids
) / len(df)

overall_entropy = sum(
    len(df[df["kmeans_cluster"] == cid]) * entropy(
        [sum(df[df["kmeans_cluster"] == cid]["Category Label"] == cat) 
         for cat in categories]
    )
    for cid in cluster_ids
) / len(df)

#append totals to the table
total_row = ["Total"] + total_counts + [round(overall_entropy, 4), round(overall_purity, 4)]
table_data.append(total_row)

#build new dataframe
colnames = ["Cluster"] + categories + ["Entropy", "Purity"]
results_df = pd.DataFrame(table_data, columns=colnames)

#print table with results
print(results_df.to_string(index=False))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


rfm=pd.read_csv("RFM_Dataset.csv")
print(rfm.head())
print(rfm.shape)
print(rfm.describe())
print(rfm.info())

## standartize the RFM Values
S=StandardScaler()
rfm_scaled=S.fit_transform(rfm[['Recency','Frequency','Monetry']])
print(rfm_scaled[:5])
print(rfm_scaled.shape)

# By using Elbow method find number of clusters of similar grroup
inertia=[]
for i in range(1,11):
    Kmeans=KMeans(n_clusters=i,random_state=42)
    Kmeans.fit(rfm_scaled)
    inertia.append(Kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11),inertia,marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Training the Final K-means model
Kmeans=KMeans(n_clusters=4,random_state=42)
rfm['Cluster']=Kmeans.fit_predict(rfm_scaled)
print(rfm.head())

# Silhouette Score to interpret the result
score=silhouette_score(rfm_scaled,rfm['Cluster'])
print("Silhouete: score:",score)

cluster_profile=rfm.groupby('Cluster')[['Recency','Frequency','Monetry']].mean()
print(cluster_profile)


## Assigning customer segement labels
cluster_labels={0:"Occasional",
                1:"At Risk",
                2:"High Value",
                3:"Regular"}


rfm['Segment']=rfm['Cluster'].map(cluster_labels)

print(rfm.head())


## Customer segment Visualizations
# 1.Recency vs Monetary
plt.figure(figsize=(10,6))
for Segment, group in rfm.groupby('Segment'):
    plt.scatter(group['Recency'],group['Monetry'], label=Segment, alpha=0.6, s=50)
plt.title("Customer Segments: Recency vs Monetary")
plt.xlabel("Recency (days)")
plt.ylabel("Monetary (Total Spend)")
plt.legend()
plt.tight_layout()
plt.show()

# 2️.Frequency vs Monetary
plt.figure(figsize=(10, 6))
for Segment, group in rfm.groupby('Segment'):
    plt.scatter(group['Frequency'], group['Monetry'],label=Segment, alpha=0.6, s=50)
plt.title("Customer Segments: Frequency vs Monetary")
plt.xlabel("Frequency (Transactions)")
plt.ylabel("Monetary (Total Spend)")
plt.legend()
plt.tight_layout()
plt.show()

# 3️⃣ 3D Plot
fig = plt.figure(figsize=(12, 8))
ax= fig.add_subplot(projection='3d')
for Segment, group in rfm.groupby('Segment'):
    ax.scatter(group['Recency'], group['Frequency'], group['Monetry'],label=Segment, alpha=0.6, s=50)
ax.set_title("3D RFM Cluster Visualization")
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary")
ax.legend()
plt.tight_layout()
plt.show()


# cluster Bar Plot
cluster_profile.index = rfm.groupby('Cluster')['Segment'].first()
cluster_profile.plot(kind='bar', figsize=(10, 6), colormap='tab10')
plt.title("Average RFM Values per Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Average Value")
plt.tight_layout()
plt.show()

# 5️. Pie Chart
segment_counts = rfm['Segment'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(segment_counts, labels=segment_counts.index,
        autopct='%1.1f%%',colors=['#2ecc71', '#3498db', '#e67e22',"#7011edc2"])
plt.title("Customer Segment Distribution")
plt.tight_layout()
plt.show()

# Save
joblib.dump(Kmeans, "kmeans_model.pkl")
joblib.dump(S, "scaler.pkl")                 
rfm.to_csv("RFM_Segmented.csv", index=False)

# Load — exact same names
loaded_model = joblib.load("kmeans_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("Verification passed! Model and scaler loaded successfully.")
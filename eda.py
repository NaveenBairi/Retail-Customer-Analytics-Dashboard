import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## extracting by calculating total price by using quantity and unitprice
df=pd.read_csv("Cleaned_Retail_data")
print(df)  
df['TotalPrice']=df['Quantity']*df['UnitPrice']
print(df.head(2))

df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
print(df['InvoiceDate'].dtype)

# Extracting other columns
df["Month"] = df["InvoiceDate"].dt.month
df["Hour"] = df["InvoiceDate"].dt.hour
print(df.head(2))

print(df.dtypes)
print(df.tail())

## Transaction Volume by Country
country_transactions = (df["Country"].value_counts().head(10).reset_index())
country_transactions.columns = ["Country", "Transactions"]
print(country_transactions)

plt.figure(figsize=(12,6))
sns.barplot(data=country_transactions,x="Country",y="Transactions",hue='Country',palette='tab10',legend=False)
plt.title("Top 10 Countries by Number of Transactions", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Number of Transactions", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top Selling Products
products=df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
plt.figure(figsize=(12,8))
sns.barplot(data=products,x='Quantity',y="Description",palette='tab10')
plt.title("Top Selling Products")
plt.tight_layout()
plt.show()

## Monthly Sales Trend
df['Month_Name']=df['InvoiceDate'].dt.month_name()

montly=df.groupby('Month_Name')['TotalPrice'].sum().reset_index()
month_order=["January","February","March","April","May","June",
             "July","August","September","October","November","December"]
montly['Month_Name']=pd.Categorical(montly['Month_Name'],categories=month_order,ordered=True)
montly=montly.sort_values("Month_Name")
plt.figure(figsize=(12,8))
sns.lineplot(data=montly,x='Month_Name',y='TotalPrice',marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# sales by hour
hour=df.groupby('Hour')['TotalPrice'].sum().reset_index()
plt.figure(figsize=(12,8))
sns.lineplot(data=hour,x='Hour',y='TotalPrice',marker='o')
plt.title("sales by hour")
plt.tight_layout()
plt.show()

# Boxplot (Outlier Detection)
plt.figure(figsize=(10,5))
sns.boxplot(x=df["TotalPrice"])
plt.title("Total Price outlier Detection")
plt.tight_layout()
plt.show()

# correlation heatmap 
plt.figure(figsize=(8,6))
corr=df[['Quantity','UnitPrice','TotalPrice']].corr()
sns.heatmap(corr,annot=True,cmap='Blues')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Inspect monetary distribution per transaction and customer
plt.figure(figsize=(10,5))
sns.histplot(df["TotalPrice"], bins=30, kde=True)
plt.title("Distribution per transaction")
plt.show()

# Per Customer
plt.figure(figsize=(10,5))
customer=df.groupby('CustomerID')['TotalPrice'].sum()
plt.xlim(0, 5000) 
sns.histplot(customer,bins=50,kde=True)
plt.title("Distribution per customer")
plt.show()








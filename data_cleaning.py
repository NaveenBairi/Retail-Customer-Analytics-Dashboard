import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print("All libraries imported sucessfully")

df=pd.read_csv("online_retail.csv")
print(df.head())

print("\nColumn Names ")
print(df.columns)

print("\nDataset shape")
print(df.shape)


print(" \nData Types")
print(df.dtypes)

print("\n Dataset info")
print(df.info())

#finding missing values in the dataset

print("/n missing values")
print(df.isnull().sum())

df=df.dropna(subset=['CustomerID'])
print(df)

print("\nMissing values in customer_Id after removing Null values")
print(df['CustomerID'].isnull().sum())


print("/n missing values")
print(df.isnull().sum())

print(df['Description'].isnull().sum())
 
print("\nDuplicate values")
print(df.duplicated().sum())

# Drop_duplicates
df=df.drop_duplicates()
print(df)

## Checking Invalid InvoiceNo Column
cancelled_voice=df[df['InvoiceNo'].astype(str).str.startswith("C")]
print(cancelled_voice.shape[0])
print(cancelled_voice.head())

print("\nCancelled_Invoice")
print(len(cancelled_voice))

print(cancelled_voice[["InvoiceNo", "Description", "Quantity", "UnitPrice", "CustomerID"]].head(10))

# Identified Removing cancelled invoices in InvoiceNo which startswith C  
# Because Cancelled or returned transactions and will be removed because they do not represent successful purchases.

df=df[~df["InvoiceNo"].astype(str).str.startswith("C")]
print("data shape after removing InValid InvoiceNo:", df.shape)

# Checking  Negative Quantity  column which is less than 0 and equal to zero
q=df[df['Quantity'] <=0]
print("Invalid Quantity:",q.shape[0])

# Remove rows where Quantity <= 0
df = df[df["Quantity"] > 0]

print("Dataset Shape after removing invalid quantities:", df.shape)

# Checking Negative UnitPrice Cloumn which is equal to 0 and less than zero
price=df[df['UnitPrice'] <=0]
print("No of Invalid Unit Prices:",price.shape[0])

# Remove Unitprices with o values
df=df[df['UnitPrice']>0]
print("Dataset Shape after removing invalid prices:", df.shape)

df.reset_index(drop=True,inplace=True)
print("Index has Sucessfully Done")

print("\nDataTypes od Dataset")
print(df.dtypes)
print(df.head())

df.to_csv("Cleaned_Retail_Data.csv",index=False)

df.columns




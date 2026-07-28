import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## extracting by calculating total price by using quantity and unitprice
df = pd.read_csv("Cleaned_Retail_Data.csv")
print(df.head(5))  
print(df.columns)
df.info()

df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
print(df['InvoiceDate'].dtype)

df['TotalPrice']=df['Quantity']*df['UnitPrice']
print(df)

df.info()

latest_date=df['InvoiceDate'].max()
print("Latest Date:",latest_date)

# Creating Rfm table
rfm=df.groupby('CustomerID').agg({
    "InvoiceDate": lambda x:(latest_date-x.max()).days,
    "InvoiceNo":"nunique",
    "TotalPrice":"sum"}).reset_index()


rfm.columns=['CustomerID','Recency','Frequency',"Monetry"]
print(rfm.head())
print(rfm.shape)

rfm.to_csv("RFM_Dataset.csv", index=False)
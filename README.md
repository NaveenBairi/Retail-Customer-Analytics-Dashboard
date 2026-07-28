#  Retail Customer Analytics Dashboard
An end-to-end Retail Customer Analytics Dashboard built using **Python, Machine Learning, and Streamlit** to analyse customer purchasing behaviour, perform customer segmentation using **RFM Analysis**, recommend similar products, and provide interactive business insights through a modern dashboard.

##  Project Overview
Retail businesses generate large volumes of transaction data every day. This project transforms raw retail data into actionable business insights by applying data cleaning, exploratory data analysis (EDA), customer segmentation, product recommendation, and interactive visualisation.
The dashboard enables businesses to identify valuable customers, understand purchasing patterns, recommend relevant products, and make data-driven business decisions.

##  Business Objective
The primary objective of this project is to help retail businesses:
- Understand customer purchasing behaviour
- Segment customers based on RFM Analysis
- Identify high-value and at-risk customers
- Recommend similar products using Machine Learning
- Monitor sales performance through an interactive dashboard
- Support data-driven marketing strategies

#  Tech Stack
| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| IDE | Visual Studio Code |
| Version Control | Git & GitHub |

#  Project Structure
Retail-Customer-Analytics-Dashboard
│
├── app.py
├── data_cleaning.py
├── eda.py
├── rfm_analysis.py
├── customer_segment.py
├── Product_recommend.py
│
├── Cleaned_Retail_Data.csv
├── RFM_Dataset.csv
├── RFM_Segmented.csv
│
├── kmeans_model.pkl
├── scaler.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

#  Project Workflow
Retail Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Feature Engineering
      │
      ▼
RFM Analysis
      │
      ▼
Customer Segmentation
(K-Means Clustering)
      │
      ▼
Product Recommendation
(Cosine Similarity)
      │
      ▼
Interactive Streamlit Dashboard
      │
      ▼
Business Insights & Recommendations

#  Data Cleaning
- Removed missing Customer IDs
- Removed duplicate transactions
- Removed cancelled invoices
- Removed invalid quantities
- Removed invalid unit prices
- Converted InvoiceDate to datetime format
- Created TotalPrice feature
- Extracted Month from InvoiceDate
- 
#  Exploratory Data Analysis
Performed detailed exploratory analysis to understand:
- Monthly Revenue Trend
- Country-wise Revenue
- Top Selling Products
- Customer Purchase Behaviour
- Product Sales Distribution
- Quantity vs Revenue Relationship
- Correlation Analysis
- Business Performance Trends

#  RFM Analysis
Customers were analysed using three important business metrics.
### Recency
Measures how recently a customer made a purchase.
### Frequency
Measures how often a customer purchases.
### Monetary
Measures the total amount spent by each customer.

These metrics help identify loyal customers and customers likely to churn.

#  Customer Segmentation
Customers were segmented using **K-Means Clustering** into four business groups.

| Segment  | Description |
|----------|-------------|
| High Value Customers | Frequent buyers with high spending |
| Regular Customers | Consistent purchasing behaviour |
| Occasional Shoppers | Moderate purchase frequency |
| At Risk Customers | Customers who haven't purchased recently |

#  Product Recommendation System
A recommendation engine was built using **Cosine Similarity** to suggest products that are similar to the selected product.

### Benefits
- Product Discovery
- Cross Selling
- Customer Experience
- Increased Sales

#  Dashboard Features
✔ KPI Cards
- Total Customers
- Total Products
- Total Transactions
- Total Revenue
 
✔ Business Visualisations
- Monthly Revenue Trend
- Country-wise Revenue
- Top Selling Products
- Correlation Heatmap

✔ Customer Analytics
- Customer Segmentation
- RFM Prediction

✔ Product Recommendation
- Similar Product Suggestions

✔ Dataset Explorer
- Interactive Dataset View

#  How to Run

## 1️ Clone the Repository
bash
git clone https://github.com/NaveenBairi/Retail-Customer-Analytics-Dashboard.git

## 2️⃣ Navigate to the Project Folder
bash
cd Retail-Customer-Analytics-Dashboard

## 3️⃣ Create Virtual Environment (Optional)
bash
python -m venv venv

## 4️⃣ Activate Virtual Environment

### Windows
bash
venv\Scripts\activate

### macOS/Linux
bash
source venv/bin/activate

##  Install Required Libraries
bash
pip install -r requirements.txt

##  Run Streamlit Dashboard
bash
streamlit run app.py

##  Open in Browser
http://localhost:8501

## 📸 Dashboard Screenshots

###  Dashboard Overview
![Dashboard Overview](images/dashboard_overview.png)
###  Customer Segmentation
![Customer Segmentation](images/customer_segmentation.png)
###  Product Recommendation
![Product Recommendation](images/product_recommendation.png)
###  Business Insights
![Business Insights](images/business_insights.png)

#  Key Business Insights

- Revenue shows strong seasonal purchasing behaviour.
- High-value customers contribute a significant share of total revenue.
- A small number of products generate most sales.
- Product recommendations can improve cross-selling opportunities.
- Customer segmentation enables personalised marketing campaigns.
- Interactive dashboards simplify business monitoring and decision-making.

#  Business Recommendations
- Retain high-value customers through loyalty programmes.
- Re-engage at-risk customers with targeted offers.
- Promote top-selling products using bundled promotions.
- Recommend similar products to increase average order value.
- Use customer segmentation for personalised marketing.
- Continuously monitor KPIs for informed business decisions.

#  Project Outcomes
- Improved customer understanding
- Effective customer segmentation
- Product recommendation system
- Interactive business dashboard
- Better decision-making through data analytics
- 
#  Author
**Bairi Naveen**
Email: bairinaveen@gmail.com
GitHub: https://github.com/NaveenBairi
LinkedIn:
#  Support
If you found this project useful, please consider giving it a **⭐ Star** on GitHub.
It helps others discover the project and supports my work.
##  License
This project is developed for **educational and portfolio purposes**.

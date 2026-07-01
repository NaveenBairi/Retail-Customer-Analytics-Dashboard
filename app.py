import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Retail Customer Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA
df = pd.read_csv("Cleaned_Retail_data")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

rfm = pd.read_csv("RFM_Segmented.csv")

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

product_similarity = joblib.load("product_similarity.pkl")
product_list = joblib.load("product_list.pkl")

# HELPER FUNCTIONS  

def kpi_card(col, emoji, label, value, color):
    """Render one colored KPI card inside a given column."""
    col.markdown(f"""
        <div style="background:{color};padding:20px;border-radius:15px;text-align:center;">
        <h1 style="margin:0;font-size:40px;">{emoji}</h1>
        <h3 style="color:#1F2937;margin:8px 0 4px 0;">{label}</h3>
        <h1 style="color:#111827;margin:0;">{value}</h1>
        </div>
        """, unsafe_allow_html=True)

def kpi_row(cards):
    """cards = list of (emoji, label, value, color) tuples -> renders as a row of columns."""
    cols = st.columns(len(cards))
    for col, (emoji, label, value, color) in zip(cols, cards):
        kpi_card(col, emoji, label, value, color)

def top_n_bar(data, group_col, value_col, agg, n, title, orientation="v", color=None):
    """Group `data` by group_col, aggregate value_col, take top n, and plot as a bar chart."""
    grouped = (
        data.groupby(group_col)[value_col]
        .agg(agg)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    color_col = color or group_col
    x, y = (value_col, group_col) if orientation == "h" else (group_col, value_col)
    fig = px.bar(
        grouped, x=x, y=y, orientation=orientation,
        color=color_col if color_col in grouped.columns else value_col,
        title=title, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    return grouped

def histogram(data, col, nbins, color, title):
    fig = px.histogram(data, x=col, nbins=nbins, color_discrete_sequence=[color], title=title)
    fig.update_layout(template="plotly_white", xaxis_title=col, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# SIDEBAR

with st.sidebar:
    st.image("https://img.icons8.com/color/240/shop.png", width=200)
    st.markdown("# 🛒 Retail Analytics")

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "🏠 Home",
            "📊 EDA Dashboard",
            "🛍 Product Recommendation",
            "👥 Customer Segmentation",
            "📈 Business Insights",
            "ℹ About"
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#F4F5F7"},
            "icon": {"color": "#374151", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px 14px",
                "border-radius": "8px",
                "color": "#374151",
                "background-color": "#FFFFFF",
            },
            "nav-link-selected": {
                "background-color": "#E74C3C",
                "color": "white",
                "font-weight": "600",
            },
        }
    )

# HOME PAGE

if selected == "🏠 Home":

    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#0F4C81,#2563EB);padding:45px;
        border-radius:20px;color:white;text-align:center;box-shadow:0px 8px 20px rgba(0,0,0,0.3);">
        <h1 style="font-size:42px;">🛒 Retail Customer Analytics Dashboard</h1>
        <h3>Customer Segmentation | Product Recommendation</h3>
        <p>Python • Streamlit • Machine Learning • Data Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.success("""
# 👋 Welcome

This dashboard helps businesses analyze customer purchasing behaviour.

### Features
✔ Exploratory Data Analysis
✔ Customer Segmentation (RFM + KMeans)
✔ Product Recommendation System
✔ Business Insights
✔ Interactive Dashboard
""")
    with col2:
        st.image("https://img.icons8.com/fluency/480/combo-chart.png", width=300)

    st.divider()

    kpi_row([
        ("👥", "Customers",    f'{df["CustomerID"].nunique():,}', "#E8F5E9"),
        ("📦", "Products",     f'{df["Description"].nunique():,}', "#E3F2FD"),
        ("🧾", "Transactions", f'{df["InvoiceNo"].nunique():,}', "#FFF3E0"),
        ("💰", "Revenue", f'${df["TotalPrice"].sum()/1000000:.2f}M', "#F3E5F5"),
    ])

    st.write("")
    st.divider()
    st.subheader("📊 Dashboard Overview")

    monthly_sales = df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum().reset_index()
    monthly_sales["InvoiceDate"] = monthly_sales["InvoiceDate"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(monthly_sales, x="InvoiceDate", y="TotalPrice",
                       title="📈 Monthly Revenue Trend", markers=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_n_bar(df, "Country", "TotalPrice", "sum", 10, "🌍 Top 10 Countries by Revenue")

    st.divider()
    top_n_bar(df, "Description", "Quantity", "sum", 10, "🏆 Top Selling Products", orientation="h")

    st.divider()
    st.subheader("📂 Explore Dataset")
    with st.expander("Click here to view dataset"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    st.subheader("📈 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Countries", df["Country"].nunique())
    col4.metric("Customers", df["CustomerID"].nunique())

    st.divider()
    st.markdown(
        """
        <div style="text-align:center;padding:20px;color:gray;">
        <h3>❤️ Made with Streamlit</h3>
        Python • Pandas • Plotly • Scikit-Learn • Machine Learning
        <hr>
        <b>Retail Customer Analytics Dashboard</b><br>
        © 2026 Bairi Naveen
        </div>
        """,
        unsafe_allow_html=True
    )

# EDA DASHBOARD

elif selected == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("Explore customer purchasing behaviour using interactive visualizations.")

    st.divider()

    # FILTER
    st.sidebar.subheader("🔍 Filter Data")
    country = st.sidebar.selectbox(
        "🌍 Select Country",
        ["All"] + sorted(df["Country"].dropna().unique())
    )
    eda_df = df.copy() if country == "All" else df[df["Country"] == country]

    # KPI CARDS
    revenue = eda_df["TotalPrice"].sum()
    customers = eda_df["CustomerID"].nunique()
    products = eda_df["Description"].nunique()
    invoices = eda_df["InvoiceNo"].nunique()

    st.subheader("📌 Dashboard Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Revenue", f"${revenue:,.2f}")
    c2.metric("👥 Customers", f"{customers:,}")
    c3.metric("📦 Products", f"{products:,}")
    c4.metric("🧾 Invoices", f"{invoices:,}")

    st.divider()

    # Revenue / Quantity Distributions
    col1, col2 = st.columns(2)
    with col1:
        histogram(eda_df, "TotalPrice", 50, "royalblue", "💰 Revenue Distribution")
    with col2:
        histogram(eda_df, "Quantity", 50, "orange", "📦 Quantity Distribution")

    st.divider()
    top_n_bar(eda_df, "CustomerID", "TotalPrice", "sum", 10, "🏆 Top 10 Customers by Revenue")

    st.divider()
    top_n_bar(eda_df, "Description", "TotalPrice", "sum", 10, "🛍️ Top 10 Products by Revenue", orientation="h")

    st.divider()
    top_n_bar(eda_df, "Country", "TotalPrice", "sum", 10, "🌍 Top 10 Countries by Revenue")

    st.divider()

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")

    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = eda_df[["Quantity", "UnitPrice", "TotalPrice"]].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.divider()

    # BUSINESS INSIGHTS
    best_country = eda_df.groupby("Country")["TotalPrice"].sum().idxmax()
    best_product = eda_df.groupby("Description")["Quantity"].sum().idxmax()

    st.subheader("📋 Business Insights")
    st.success(f"""
### 📈 Key Insights

💰 **Total Revenue:** ${revenue:,.2f}

👥 **Total Customers:** {customers:,}

📦 **Total Products:** {products:,}

🧾 **Total Invoices:** {invoices:,}

🌍 **Highest Revenue Country:** {best_country}

🏆 **Best Selling Product:** {best_product}
""")

    st.divider()

    # FILTERED DATA
    with st.expander("📂 View Filtered Dataset"):
        st.dataframe(eda_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Filtered Data",
        eda_df.to_csv(index=False),
        file_name="Filtered_EDA_Data.csv",
        mime="text/csv"
    )

# PRODUCT RECOMMENDATION

elif selected == "🛍 Product Recommendation":

    st.title("🛍 Product Recommendation System")

    st.markdown("""
Find similar products using **Item-Based Collaborative Filtering**.

Select any product below and the system will recommend the **Top 5 similar products**.
""")

    st.divider()

    # Product Selection
    product_name = st.selectbox(
        "📦 Select Product",
        sorted(product_list)
    )

    st.write("")

    if st.button("🚀 Recommend Products"):

        if product_name in product_similarity.index:

            recommendations = (
                product_similarity[product_name]
                .sort_values(ascending=False)
                .iloc[1:6]
            )

            st.success(f"Top recommendations for **{product_name}**")

            st.write("")

            cols = st.columns(5)

            for col, product in zip(cols, recommendations.index):

                with col:

                    st.markdown(f"""
                    <div style="
                    background:#E3F2FD;
                    padding:20px;
                    border-radius:15px;
                    text-align:center;
                    height:180px;
                    ">

                    <h1>📦</h1>

                    <h4>{product}</h4>

                    </div>
                    """,
                    unsafe_allow_html=True)

        else:

            st.error("Product not found.")

    st.divider()

    st.subheader("📖 How Recommendations Work")

    st.info("""
This recommendation engine uses **Item-Based Collaborative Filtering**.

• Finds products frequently purchased together.

• Calculates similarity between products.

• Recommends the Top 5 most similar products.

This helps businesses improve cross-selling and customer experience.
""")

    st.divider()

    with st.expander("📂 View Available Products"):

        st.dataframe(
            pd.DataFrame(
                sorted(product_list),
                columns=["Products"]
            ),
            use_container_width=True
        )

    st.download_button(
        "⬇️ Download Product List",
        pd.DataFrame(sorted(product_list), columns=["Products"]).to_csv(index=False),
        file_name="Product_List.csv",
        mime="text/csv"
    )

# CUSTOMER SEGMENTATION

elif selected == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.markdown("""
Predict customer segments using the trained **KMeans Machine Learning Model**
based on **Recency, Frequency and Monetary (RFM)** values.
""")
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "📅 Recency (Days)",
            min_value=0,
            value=30
        )

    with col2:
        frequency = st.number_input(
            "🛒 Frequency",
            min_value=1,
            value=5
        )

    with col3:
        monetary = st.number_input(
            "💰 Monetary",
            min_value=0.0,
            value=1000.0
        )

    st.write("")

    if st.button("🚀 Predict Segment"):

        sample = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(sample)[0]
        labels = {
            0: "🟢 Occasional Customer",
            1: "🔴 At Risk Customer",
            2: "🟣 High Value Customer",
            3: "🟡 Regular Customer"
        }

        colors = {
            0: "#2ECC71",
            1: "#E74C3C",
            2: "#8E44AD",
            3: "#F39C12"
        }

        st.markdown(f"""
        <div style="
        background:{colors[cluster]};
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        font-size:24px;
        box-shadow:0px 5px 15px rgba(0,0,0,0.3);
        ">

        <h2>{labels[cluster]}</h2>

        </div>
        """, unsafe_allow_html=True)

        if cluster == 2:
            st.success("⭐ High Value Customer — Offer premium membership, loyalty rewards and exclusive offers.")

        elif cluster == 3:
            st.info("👍 Regular Customer — Recommend related products and seasonal discounts.")

        elif cluster == 0:
            st.warning("🙂 Occasional Customer — Encourage repeat purchases with coupons and email campaigns.")

        else:
            st.error("⚠️ At Risk Customer — Launch win-back campaigns and personalized discounts.")

    st.divider()
    st.subheader("📊 Customer Segment Distribution")

    segment_count = (
        rfm["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_count.columns = ["Segment", "Customers"]
## pie chart
    fig = px.pie(
        segment_count,
        names="Segment",
        values="Customers",
        hole=0.45,
        color="Segment",
        title="Customer Segment Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    st.subheader("📈 RFM Customer Segments")
#  Scatter plot
    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetry",
        size="Frequency",
        color="Segment",
        hover_data=["CustomerID"],
        title="Customer Segmentation using RFM"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # SUMMARY TABLE

    st.subheader("📋 Segment Summary")

    summary = (
        rfm.groupby("Segment")
        [["Recency", "Frequency", "Monetry"]]
        .mean()
        .round(2)
    )

    st.dataframe(summary, use_container_width=True)

    st.divider()

    # DATASET
    
    with st.expander("📂 View Customer Segmentation Dataset"):

        st.dataframe(
            rfm,
            use_container_width=True
        )

    st.download_button(
        "⬇️ Download Dataset",
        rfm.to_csv(index=False),
        file_name="RFM_Segmented.csv",
        mime="text/csv"
    )

    st.divider()

    # BUSINESS RECOMMENDATIONS
   
    st.subheader("💡 Business Recommendations")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
### 🟣 High Value Customers

      ✅ Loyalty Rewards

      ✅ VIP Membership

      ✅ Early Access

      ✅ Premium Products
      """)

        st.info("""
        🟡 Regular Customers

        ✅ Cross Selling

        ✅ Personalized Recommendations

        ✅ Seasonal Discounts
        """)

    with col2:

        st.warning("""
        ### 🟢 Occasional Customers

        ✅ Email Campaigns

        ✅ Coupons

        ✅ Festival Offers
        """)

        st.error("""
        ### 🔴 At Risk Customers

        ✅ Win-back Campaigns

        ✅ Cashback

        ✅ Reminder Emails

        ✅ Personalized Discounts
        """)
# BUSINESS INSIGHTS


elif selected == "📈 Business Insights":

    st.markdown("## Business Insights")

    total_revenue = df["TotalPrice"].sum()
    total_orders = df["InvoiceNo"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0
    repeat_customers = df.groupby("CustomerID")["InvoiceNo"].nunique().gt(1).sum()

    best_country = df.groupby("Country")["TotalPrice"].sum().idxmax()
    best_product = df.groupby("Description")["Quantity"].sum().idxmax()
    top_segment = (
        rfm.groupby("Segment")["Monetry"].sum().sort_values(ascending=False).index[0]
    )

    st.markdown(f"**Total Revenue:** ${total_revenue:,.0f}")
    st.markdown(f"**Total Orders:** {total_orders:,}")
    st.markdown(f"**Average Order Value:** ${avg_order_value:,.2f}")
    st.markdown(f"**Repeat Customers:** {repeat_customers:,}")
    st.markdown(f"**Top Revenue Country:** {best_country}")
    st.markdown(f"**Best Selling Product:** {best_product}")
    st.markdown(f"**Highest Value Segment:** {top_segment}")

    st.write("")

    monthly_sales = (
        df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"]
        .sum()
        .reset_index()
    )
    monthly_sales["InvoiceDate"] = monthly_sales["InvoiceDate"].astype(str)

    fig = px.line(
        monthly_sales, x="InvoiceDate", y="TotalPrice",
        title="Monthly Revenue Trend", template="plotly_white"
    )
    fig.update_traces(line_color="#E74C3C")
    st.plotly_chart(fig, use_container_width=True)

# ABOUT

elif selected == "ℹ About":

    st.title("ℹ️ About the Project")

    st.markdown("""
## 🛒 Retail Customer Analytics Dashboard

The **Retail Customer Analytics Dashboard** is an end-to-end data analytics application developed to help retail businesses understand customer purchasing behavior, identify valuable customers, recommend similar products, and generate actionable business insights.

This project combines **Data Analysis, Machine Learning, and Interactive Visualization** into a single dashboard built using Streamlit.
""")

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
- 📊 Perform Exploratory Data Analysis (EDA)
- 👥 Segment customers using RFM Analysis and KMeans Clustering
- 🛍️ Recommend similar products using Collaborative Filtering
- 📈 Generate business insights through interactive dashboards
- 💡 Support data-driven business decision making
""")

    st.divider()

    st.subheader("🛠️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### Programming
- Python
- Pandas
- NumPy
- Scikit-Learn
        """)

    with col2:
        st.markdown("""
### Visualization
- Streamlit
- Plotly
- Matplotlib
- Seaborn
        """)

    st.divider()

    st.subheader("📂 Dashboard Modules")

    st.markdown("""
🏠 **Home**
- Dashboard Overview
- KPI Cards
- Revenue Trends
- Dataset Summary

📊 **EDA Dashboard**
- Revenue Distribution
- Customer Analysis
- Product Analysis
- Country Analysis
- Correlation Heatmap

👥 **Customer Segmentation**
- RFM Analysis
- KMeans Prediction
- Segment Distribution
- Business Recommendations

🛍️ **Product Recommendation**
- Similar Product Recommendation
- Collaborative Filtering

📈 **Business Insights**
- Revenue KPIs
- Top Products
- Top Countries
- Customer Insights
""")

    st.divider()

    st.subheader("🚀 Key Features")

    st.success("""
✅ Interactive Dashboard

✅ Machine Learning Prediction

✅ Product Recommendation System

✅ Customer Segmentation

✅ Business Insights

✅ Downloadable Reports
""")

    st.divider()

    st.subheader("👨‍💻 Developed By")

    st.info("""
**Bairi Naveen**

Aspiring Data Analyst | Data Science Enthusiast

Skills:
- Excel
- SQL
- Python
- Power BI
- Streamlit
- Machine Learning
""")

    st.divider()

    st.markdown(
    """
    <div style="text-align:center;padding:20px;color:gray;">

    <h3>Retail Customer Analytics Dashboard</h3>

    Python • Streamlit • Machine Learning • Plotly • Scikit-Learn

    <br><br>

    © 2026 Bairi Naveen

    </div>
    """,
    unsafe_allow_html=True
    )
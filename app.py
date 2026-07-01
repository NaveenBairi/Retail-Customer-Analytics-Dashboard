import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Retail Customer Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
# STATIC CONFIG (single source of truth — was duplicated
# across the Segmentation page in the original file)

SEGMENT_INFO = {
    0: {
        "label": "🟢 Occasional Customer",
        "color": "#2ECC71",
        "box": "warning",
        "advice": "🙂 Occasional Customer — Encourage repeat purchases with coupons and email campaigns.",
        "recs": ["Email Campaigns", "Coupons", "Festival Offers"],
    },
    1: {
        "label": "🔴 At Risk Customer",
        "color": "#E74C3C",
        "box": "error",
        "advice": "⚠️ At Risk Customer — Launch win-back campaigns and personalized discounts.",
        "recs": ["Win-back Campaigns", "Cashback", "Reminder Emails", "Personalized Discounts"],
    },
    2: {
        "label": "🟣 High Value Customer",
        "color": "#8E44AD",
        "box": "success",
        "advice": "⭐ High Value Customer — Offer premium membership, loyalty rewards and exclusive offers.",
        "recs": ["Loyalty Rewards", "VIP Membership", "Early Access", "Premium Products"],
    },
    3: {
        "label": "🟡 Regular Customer",
        "color": "#F39C12",
        "box": "info",
        "advice": "👍 Regular Customer — Recommend related products and seasonal discounts.",
        "recs": ["Cross Selling", "Personalized Recommendations", "Seasonal Discounts"],
    },
}

BOX_FN = {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}

FOOTER_HTML = """
<div style="text-align:center;padding:20px;color:gray;">
<h3>{heading}</h3>
Python • Streamlit • Machine Learning • Plotly • Scikit-Learn
<hr>
<b>Retail Customer Analytics Dashboard</b><br>
© 2026 Bairi Naveen
</div>
"""
# DATA / MODEL LOADING 

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Retail_Data_csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    rfm = pd.read_csv("RFM_Segmented.csv")
    return df, rfm


@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    product_similarity = joblib.load("product_similarity.pkl")
    product_list = joblib.load("product_list.pkl")
    return kmeans, scaler, product_similarity, product_list


df, rfm = load_data()
kmeans, scaler, product_similarity, product_list = load_models()

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


def metric_row(items):
    """items = list of (label, value) tuples -> renders as a row of st.metric."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


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


def colored_box(text, color, text_color="white", font_size="24px"):
    """Generic colored callout div (used for segment-prediction result)."""
    st.markdown(f"""
        <div style="background:{color};padding:25px;border-radius:15px;text-align:center;
        color:{text_color};font-size:{font_size};box-shadow:0px 5px 15px rgba(0,0,0,0.3);">
        <h2>{text}</h2>
        </div>
        """, unsafe_allow_html=True)


def download_csv_button(data, label, file_name):
    st.download_button(label, data.to_csv(index=False), file_name=file_name, mime="text/csv")


def business_summary(data, rfm_data=None):
    """Compute the recurring set of headline business metrics from a (filtered) dataframe."""
    revenue = data["TotalPrice"].sum()
    customers = data["CustomerID"].nunique()
    products = data["Description"].nunique()
    invoices = data["InvoiceNo"].nunique()
    best_country = data.groupby("Country")["TotalPrice"].sum().idxmax()
    best_product = data.groupby("Description")["Quantity"].sum().idxmax()

    summary = {
        "revenue": revenue,
        "customers": customers,
        "products": products,
        "invoices": invoices,
        "best_country": best_country,
        "best_product": best_product,
    }

    if rfm_data is not None:
        summary["top_segment"] = (
            rfm_data.groupby("Segment")["Monetry"].sum().sort_values(ascending=False).index[0]
        )
    return summary


def monthly_revenue_chart(data, title="Monthly Revenue Trend", line_color=None):
    monthly = data.groupby(data["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum().reset_index()
    monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
    fig = px.line(monthly, x="InvoiceDate", y="TotalPrice", title=title,
                  markers=True, template="plotly_white")
    if line_color:
        fig.update_traces(line_color=line_color)
    st.plotly_chart(fig, use_container_width=True)


def hero_banner(gradient, title, subtitle, tagline):
    st.markdown(f"""
        <div style="background:{gradient};padding:45px;
        border-radius:20px;color:white;text-align:center;box-shadow:0px 8px 20px rgba(0,0,0,0.3);">
        <h1 style="font-size:42px;">{title}</h1>
        <h3>{subtitle}</h3>
        <p>{tagline}</p>
        </div>
        """, unsafe_allow_html=True)


def footer(heading):
    st.markdown(FOOTER_HTML.format(heading=heading), unsafe_allow_html=True)

# SIDEBAR / NAVIGATION
PAGES = ["🏠 Home", "📊 EDA Dashboard", "🛍 Product Recommendation",
         "👥 Customer Segmentation", "📈 Business Insights", "ℹ About"]

with st.sidebar:
    st.image("https://img.icons8.com/color/240/shop.png", width=200)
    st.markdown("# 🛒 Retail Analytics")

    selected = option_menu(
        menu_title="Navigation",
        options=PAGES,
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#F4F5F7"},
            "icon": {"color": "#374151", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px", "text-align": "left", "margin": "4px 0",
                "padding": "10px 14px", "border-radius": "8px",
                "color": "#374151", "background-color": "#FFFFFF",
            },
            "nav-link-selected": {"background-color": "#E74C3C", "color": "white", "font-weight": "600"},
        }
    )

# HOME PAGE

if selected == "🏠 Home":

    hero_banner(
        "linear-gradient(90deg,#0F4C81,#2563EB)",
        "🛒 Retail Customer Analytics Dashboard",
        "Customer Segmentation | Product Recommendation",
        "Python • Streamlit • Machine Learning • Data Analytics",
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
        ("👥", "Customers", f'{df["CustomerID"].nunique():,}', "#E8F5E9"),
        ("📦", "Products", f'{df["Description"].nunique():,}', "#E3F2FD"),
        ("🧾", "Transactions", f'{df["InvoiceNo"].nunique():,}', "#FFF3E0"),
        ("💰", "Revenue", f'${df["TotalPrice"].sum()/1_000_000:.2f}M', "#F3E5F5"),
    ])

    st.write("")
    st.divider()
    st.subheader("📊 Dashboard Overview")

    col1, col2 = st.columns(2)
    with col1:
        monthly_revenue_chart(df, "📈 Monthly Revenue Trend")
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
    metric_row([
        ("Rows", f"{df.shape[0]:,}"),
        ("Columns", df.shape[1]),
        ("Countries", df["Country"].nunique()),
        ("Customers", df["CustomerID"].nunique()),
    ])

    st.divider()
    footer("❤️ Made with Streamlit")

# EDA DASHBOARD

elif selected == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("Explore customer purchasing behaviour using interactive visualizations.")
    st.divider()

    st.sidebar.subheader("🔍 Filter Data")
    country = st.sidebar.selectbox("🌍 Select Country", ["All"] + sorted(df["Country"].dropna().unique()))
    eda_df = df.copy() if country == "All" else df[df["Country"] == country]

    summary = business_summary(eda_df)

    st.subheader("📌 Dashboard Summary")
    metric_row([
        ("💰 Revenue", f'${summary["revenue"]:,.2f}'),
        ("👥 Customers", f'{summary["customers"]:,}'),
        ("📦 Products", f'{summary["products"]:,}'),
        ("🧾 Invoices", f'{summary["invoices"]:,}'),
    ])

    st.divider()
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
    st.subheader("🔥 Correlation Heatmap")
    corr = eda_df[["Quantity", "UnitPrice", "TotalPrice"]].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.divider()
    st.subheader("📋 Business Insights")
    st.success(f"""
### 📈 Key Insights

💰 **Total Revenue:** ${summary['revenue']:,.2f}

👥 **Total Customers:** {summary['customers']:,}

📦 **Total Products:** {summary['products']:,}

🧾 **Total Invoices:** {summary['invoices']:,}

🌍 **Highest Revenue Country:** {summary['best_country']}

🏆 **Best Selling Product:** {summary['best_product']}
""")

    st.divider()
    with st.expander("📂 View Filtered Dataset"):
        st.dataframe(eda_df, use_container_width=True)
    download_csv_button(eda_df, "⬇️ Download Filtered Data", "Filtered_EDA_Data.csv")

# PRODUCT RECOMMENDATION

elif selected == "🛍 Product Recommendation":

    st.title("🛍 Product Recommendation System")
    st.markdown("""
Find similar products using **Item-Based Collaborative Filtering**.

Select any product below and the system will recommend the **Top 5 similar products**.
""")
    st.divider()

    product_name = st.selectbox("📦 Select Product", sorted(product_list))
    st.write("")

    if st.button("🚀 Recommend Products"):
        if product_name in product_similarity.index:
            recommendations = product_similarity[product_name].sort_values(ascending=False).iloc[1:6]
            st.success(f"Top recommendations for **{product_name}**")
            st.write("")

            cols = st.columns(5)
            for col, product in zip(cols, recommendations.index):
                with col:
                    st.markdown(f"""
                    <div style="background:#E3F2FD;padding:20px;border-radius:15px;
                    text-align:center;height:180px;">
                    <h1>📦</h1>
                    <h4>{product}</h4>
                    </div>
                    """, unsafe_allow_html=True)
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
        st.dataframe(pd.DataFrame(sorted(product_list), columns=["Products"]), use_container_width=True)
    download_csv_button(
        pd.DataFrame(sorted(product_list), columns=["Products"]),
        "⬇️ Download Product List", "Product_List.csv"
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
        recency = st.number_input("📅 Recency (Days)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("🛒 Frequency", min_value=1, value=5)
    with col3:
        monetary = st.number_input("💰 Monetary", min_value=0.0, value=1000.0)

    st.write("")

    if st.button("🚀 Predict Segment"):
        sample = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(sample)[0]
        info = SEGMENT_INFO[cluster]

        colored_box(info["label"], info["color"])
        BOX_FN[info["box"]](info["advice"])

    st.divider()
    st.subheader("📊 Customer Segment Distribution")
    segment_count = rfm["Segment"].value_counts().reset_index()
    segment_count.columns = ["Segment", "Customers"]
    fig = px.pie(segment_count, names="Segment", values="Customers", hole=0.45,
                 color="Segment", title="Customer Segment Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📈 RFM Customer Segments")
    fig = px.scatter(rfm, x="Recency", y="Monetry", size="Frequency", color="Segment",
                      hover_data=["CustomerID"], title="Customer Segmentation using RFM")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📋 Segment Summary")
    summary_table = rfm.groupby("Segment")[["Recency", "Frequency", "Monetry"]].mean().round(2)
    st.dataframe(summary_table, use_container_width=True)

    st.divider()
    with st.expander("📂 View Customer Segmentation Dataset"):
        st.dataframe(rfm, use_container_width=True)
    download_csv_button(rfm, "⬇️ Download Dataset", "RFM_Segmented.csv")

    st.divider()
    st.subheader("💡 Business Recommendations")

    # Driven by SEGMENT_INFO instead of four hand-written blocks
    box_order = [2, 3, 0, 1]  # High value, Regular, Occasional, At risk
    col1, col2 = st.columns(2)
    for idx, cluster_id in enumerate(box_order):
        info = SEGMENT_INFO[cluster_id]
        rec_lines = "\n\n".join(f"✅ {r}" for r in info["recs"])
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            BOX_FN[info["box"]](f"### {info['label']}\n\n{rec_lines}")

# BUSINESS INSIGHTS

elif selected == "📈 Business Insights":

    st.markdown("## Business Insights")

    summary = business_summary(df, rfm)
    repeat_customers = df.groupby("CustomerID")["InvoiceNo"].nunique().gt(1).sum()

    st.markdown(f"**Total Revenue:** ${summary['revenue']:,.0f}")
    st.markdown(f"**Total Orders:** {summary['invoices']:,}")
    st.markdown(f"**Average Order Value:** ${summary['revenue'] / summary['invoices']:,.2f}" if summary['invoices'] else "**Average Order Value:** $0.00")
    st.markdown(f"**Repeat Customers:** {repeat_customers:,}")
    st.markdown(f"**Top Revenue Country:** {summary['best_country']}")
    st.markdown(f"**Best Selling Product:** {summary['best_product']}")
    st.markdown(f"**Highest Value Segment:** {summary['top_segment']}")

    st.write("")
    monthly_revenue_chart(df, "Monthly Revenue Trend", line_color="#E74C3C")

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
        st.markdown("### Programming\n- Python\n- Pandas\n- NumPy\n- Scikit-Learn")
    with col2:
        st.markdown("### Visualization\n- Streamlit\n- Plotly\n- Matplotlib\n- Seaborn")

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
    footer("Retail Customer Analytics Dashboard")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/data.csv", encoding="ISO-8859-1")

# clean data
df = df.drop_duplicates().dropna(subset=["CustomerID"])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df[(~df["InvoiceNo"].astype(str).str.startswith("C")) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

# Aggregations & Categorizations
country_sales = df.groupby("Country")["TotalSales"].sum().sort_values(ascending=False)
top_products = df.groupby("Description")["TotalSales"].sum().sort_values(ascending=False).head(10)

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
monthly_sales = df.groupby("Month")["TotalSales"].sum()

days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_sales = df.groupby(df["InvoiceDate"].dt.day_name())["TotalSales"].sum().reindex(days_order)

df["Hour"] = df["InvoiceDate"].dt.hour
hour_sales = df.groupby("Hour")["TotalSales"].sum()

customer_sales = df.groupby("CustomerID")["TotalSales"].sum().sort_values(ascending=False)

# Visualizations
plt.figure(figsize=(10, 5))
country_sales.head(10).plot(kind="bar", title="Top 10 Countries by Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
top_products.sort_values().plot(kind="barh", title="Top 10 Products by Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Sales")
plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
day_sales.plot(kind="bar", title="Sales by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
hour_sales.plot(kind="line", marker="o", title="Sales by Hour")
plt.grid()
plt.tight_layout()
plt.show()

#  RFM Analysis
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalSales": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

def customer_segment(score):
    if score >= 10: return "Best Customers"
    elif score >= 7: return "Loyal Customers"
    elif score >= 5: return "Potential Customers"
    else: return "Low Activity"

rfm["Segment"] = rfm["RFM_Score"].apply(customer_segment)
segment_sales = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))
segment_sales.plot(kind="bar", title="Sales by Customer Segment")
plt.xticks(rotation=30) 
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    rfm[["Recency", "Frequency", "Monetary"]].corr(),
    annot=True,
    cmap="Blues"
)

plt.title("RFM Correlation")
plt.tight_layout()
plt.show()

# 5. Final Summary Output
print("\n========== PROJECT SUMMARY ==========")
print("Total Sales:", round(df["TotalSales"].sum(), 2))
print("Total Orders:", df["InvoiceNo"].nunique())
print("Total Customers:", df["CustomerID"].nunique())
print("Total Products:", df["StockCode"].nunique())
print("Average Order Value:", round(df.groupby("InvoiceNo")["TotalSales"].sum().mean(), 2))
print("Best Country:", country_sales.index[0])
print("Best Selling Product:", top_products.index[0])
print("Best Customer Segment:", segment_sales.index[0])
print("=====================================")
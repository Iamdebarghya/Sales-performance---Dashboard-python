import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset with "North," "South," "East," and "West" regions
data = pd.DataFrame({
    "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08"],
    "Region": ["North", "South", "East", "West", "North", "South", "East", "West"],
    "Product": ["Product A", "Product B", "Product C", "Product D", "Product B", "Product A", "Product A", "Product B"],
    "Sales": [1000, 1200, 800, 900, 1100, 1300, 1500, 1400]
})

# Display the full dataset to verify
print("Full Dataset:")
print(data)

# Simulate user-selected filters for regions and products
region_filter = ["North", "South", "East", "West"]  # Include all regions
product_filter = ["Product A", "Product B", "Product C", "Product D"]  # Include all products

# Apply the filters to the dataset
filtered_data = data[
    (data["Region"].isin(region_filter)) & 
    (data["Product"].isin(product_filter))
]

# Display the filtered dataset
print("\nFiltered Data:")
print(filtered_data)

# Visualize Sales by Region (Bar Chart)
print("\nSales by Region (Bar Chart):")
region_sales = filtered_data.groupby("Region")["Sales"].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="Region", y="Sales", data=region_sales)
plt.title("Sales by Region (Bar Chart)")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()

# Visualize Sales by Region (Pie Chart)
print("\nSales by Region (Pie Chart):")
plt.figure(figsize=(8, 8))
plt.pie(region_sales["Sales"], labels=region_sales["Region"], autopct="%1.1f%%", startangle=140)
plt.title("Sales by Region (Pie Chart)")
plt.show()

# Visualize Sales by Product (Bar Chart)
print("\nSales by Product (Bar Chart):")
product_sales = filtered_data.groupby("Product")["Sales"].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="Product", y="Sales", data=product_sales)
plt.title("Sales by Product (Bar Chart)")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.show()

# Visualize Sales by Product (Pie Chart)
print("\nSales by Product (Pie Chart):")
plt.figure(figsize=(8, 8))
plt.pie(product_sales["Sales"], labels=product_sales["Product"], autopct="%1.1f%%", startangle=140)
plt.title("Sales by Product (Pie Chart)")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from google.colab import files

uploaded = files.upload()

complete_data = pd.read_csv('synthetic_retail_sales.csv')
hidden_data = pd.read_csv('synthetic_retail_sales_hidden.csv')

print(" Files loaded successfully!")

def data_overview(df, dataset_name):
    print(f"\n{'='*50}")
    print(f"EDA - {dataset_name}")
    print(f"{'='*50}\n")
    
    print("Data sample (first 5 rows):")
    display(df.head())
    
    print("\nData info:")
    display(df.info())
    
    print("\nNumerical statistics:")
    display(df.describe().transpose())
    
    print("\nMissing values:")
    missing_data = df.isnull().sum().to_frame(name='Missing Values')
    missing_data['Missing %'] = round((missing_data['Missing Values'] / len(df)) * 100, 2)
    display(missing_data[missing_data['Missing Values'] > 0])

data_overview(complete_data, "Complete Data")
data_overview(hidden_data, "Data with Missing Values")

def product_category_analysis(df):
    print("\nProduct Category Analysis:")
    
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='ProductCategory', order=df['ProductCategory'].value_counts().index)
    plt.title('Product Category Distribution')
    plt.xlabel('Product Category')
    plt.ylabel('Transaction Count')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()
    
    print("\nTotal Sales by Category:")
    sales_by_category = df.groupby('ProductCategory')['TotalSale'].sum().sort_values(ascending=False)
    display(sales_by_category.to_frame())
    
    plt.figure(figsize=(12, 6))
    sales_by_category.plot(kind='bar')
    plt.title('Total Sales by Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

product_category_analysis(complete_data)

def store_location_analysis(df):
    print("\nStore Location Analysis:")
    
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='StoreLocation', order=df['StoreLocation'].value_counts().index)
    plt.title('Transactions by Location')
    plt.xlabel('Store Location')
    plt.ylabel('Transaction Count')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()
    
    print("\nTotal Sales by Location:")
    sales_by_location = df.groupby('StoreLocation')['TotalSale'].sum().sort_values(ascending=False)
    display(sales_by_location.to_frame())
    
    plt.figure(figsize=(12, 6))
    sales_by_location.plot(kind='bar')
    plt.title('Total Sales by Store Location')
    plt.xlabel('Store Location')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

store_location_analysis(complete_data)

def payment_method_analysis(df):
    print("\nPayment Method Analysis:")
    
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='PaymentMethod', order=df['PaymentMethod'].value_counts().index)
    plt.title('Payment Methods Distribution')
    plt.xlabel('Payment Method')
    plt.ylabel('Transaction Count')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()
    
    print("\nAverage Sales by Payment Method:")
    avg_sales_by_payment = df.groupby('PaymentMethod')['TotalSale'].mean().sort_values(ascending=False)
    display(avg_sales_by_payment.to_frame())

payment_method_analysis(complete_data)

def return_analysis(df):
    print("\nReturn Rate Analysis:")
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='Returned')
    plt.title('Return Status Distribution')
    plt.xlabel('Return Status')
    plt.ylabel('Transaction Count')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()
    
    print("\nReturn Rate by Product Category:")
    return_rate_by_category = df.groupby('ProductCategory')['Returned'].apply(
        lambda x: (x == 'Yes').sum() / x.count() * 100
    ).sort_values(ascending=False)
    display(return_rate_by_category.to_frame(name='Return Rate (%)'))

return_analysis(complete_data)

def variable_relationships(df):
    print("\nNumerical Variables Relationships:")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Numerical Variables Correlation Matrix')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='UnitPrice', y='QuantitySold', hue='ProductCategory')
    plt.title('Unit Price vs Quantity Sold')
    plt.xlabel('Unit Price')
    plt.ylabel('Quantity Sold')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='CustomerAge', y='TotalSale', hue='ProductCategory')
    plt.title('Customer Age vs Total Sales')
    plt.xlabel('Customer Age')
    plt.ylabel('Total Sales')
    plt.show()

variable_relationships(complete_data)

def handle_missing_data(df):
    print("\nHandling Missing Data:")
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    num_imputer = SimpleImputer(strategy='mean')
    df_clean[numeric_cols] = num_imputer.fit_transform(df_clean[numeric_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])
    
    print("\nMissing Values After Imputation:")
    display(df_clean.isnull().sum().to_frame(name='Missing Values'))
    
    return df_clean

cleaned_data = handle_missing_data(hidden_data)

def compare_datasets(original, cleaned):
    print("\nOriginal vs Cleaned Data Comparison:")
    
    print("\nNumerical Columns Comparison:")
    numeric_cols = original.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        print(f"\nColumn: {col}")
        compare_df = pd.DataFrame({
            'Original Data': original[col].describe(),
            'Cleaned Data': cleaned[col].describe()
        })
        display(compare_df)

compare_datasets(complete_data, cleaned_data)

def prepare_for_knn(df):
    print("\nPreparing Data for KNN:")
    
    df_prepared = df.copy()
    categorical_cols = ['ProductCategory', 'StoreLocation', 'PaymentMethod', 'Returned']
    df_prepared = pd.get_dummies(df_prepared, columns=categorical_cols, drop_first=True)
    
    print("\nData Sample After Encoding:")
    display(df_prepared.head())
    
    return df_prepared

complete_prepared = prepare_for_knn(complete_data)
cleaned_prepared = prepare_for_knn(cleaned_data)

print("\nEDA completed successfully!")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
import matplotlib.dates as mdates
from matplotlib.widgets import Cursor

# Load the dataset
df = pd.read_csv('transactions_train.csv')

# Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%m-%Y %H:%M:%S')

# Function for Rule-based Fraud Detection (with more rules)
def apply_rule_based_fraud_rules(df):
    # Rule 1: Flag if transaction amount is above 1 lakh
    df['is_fraud_predicted_rule'] = df['transaction_amount'] > 100000

    # Rule 2: Flag if transaction happens between 12 AM and 4 AM
    df['is_fraud_predicted_rule'] |= (df['transaction_date'].dt.hour >= 0) & (df['transaction_date'].dt.hour < 4)

    # Rule 3: Flag if payer or payee is anonymous (assuming no logic to judge anonymity here)
    # We won't flag anything as anonymous, as you said not to do it automatically based on column name
    # Rule 4: High frequency of transactions by the same payer or payee in a short period
    df['is_fraud_predicted_rule'] |= df.groupby('payee_id_anonymous')['transaction_date'].transform('count') > 5

    # Rule 5: Transactions with an unusually high frequency from a specific channel
    df['is_fraud_predicted_rule'] |= df.groupby('transaction_channel')['transaction_date'].transform('count') > 10

    # Rule 6: Unusual transaction amounts compared to the historical average
    df['average_transaction_amount'] = df.groupby('payee_id_anonymous')['transaction_amount'].transform('mean')
    df['is_fraud_predicted_rule'] |= (df['transaction_amount'] > 3 * df['average_transaction_amount'])

    return df

# Apply rule-based fraud detection
df = apply_rule_based_fraud_rules(df)

# Unsupervised AI-based Fraud Detection using Isolation Forest
def apply_ai_based_fraud_detection(df):
    # Feature Engineering for Unsupervised Model
    df['transaction_hour'] = df['transaction_date'].dt.hour
    df['transaction_day'] = df['transaction_date'].dt.dayofweek
    df['transaction_amount_log'] = np.log1p(df['transaction_amount'])  # Log-transformed to reduce skewness

    features = ['transaction_amount_log', 'transaction_hour', 'transaction_day', 'transaction_channel']
    
    # Apply Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    df['is_fraud_predicted_ai'] = isolation_forest.fit_predict(df[features])
    df['is_fraud_predicted_ai'] = df['is_fraud_predicted_ai'].apply(lambda x: True if x == -1 else False)  # Convert to True for fraud, False for not

    return df

# Apply unsupervised AI-based fraud detection
df = apply_ai_based_fraud_detection(df)

# Function to filter data by transaction IDs
def filter_data(df, transaction_ids):
    filtered_df = df[df['transaction_id_anonymous'].isin(transaction_ids)]
    return filtered_df

# Function to display fraud detection for all transactions
def view_all_transactions_fraud_detection():
    print("Displaying fraud detection results for all transactions:")
    print(df[['transaction_id_anonymous', 'is_fraud_predicted_rule', 'is_fraud_predicted_ai', 'is_fraud']])

# Function to process and evaluate batch transactions
def evaluate_batch_transactions(batch_size):
    transaction_ids = []
    for i in range(batch_size):
        transaction_id = input(f"Enter Transaction ID {i + 1}: ")
        transaction_ids.append(transaction_id)
    
    # Filter data based on input transaction IDs
    selected_transactions = filter_data(df, transaction_ids)
    print("\nSelected transactions for fraud detection:")
    print(selected_transactions[['transaction_id_anonymous', 'is_fraud_predicted_rule', 'is_fraud_predicted_ai', 'is_fraud']])

    # Plotting Confusion Matrix for AI-based predictions (using reported fraud as ground truth)
    def plot_confusion_matrix(y_true, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix using Seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Confusion Matrix for AI-based Fraud Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # Plot confusion matrix (AI model vs Reported fraud)
    plot_confusion_matrix(selected_transactions['is_fraud'], selected_transactions['is_fraud_predicted_ai'])

    # Evaluate Model: Precision and Recall
    def evaluate_model(y_true, y_pred):
        # Precision and Recall
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print("\nEvaluation Metrics for AI-based model:")
        print("Precision:", precision)
        print("Recall:", recall)

    # Call evaluation function for the AI model
    evaluate_model(selected_transactions['is_fraud'], selected_transactions['is_fraud_predicted_ai'])

# Plot time series graph for fraud predictions
def plot_time_series_graph():
    plt.figure(figsize=(10, 6))
    # Plot for Rule-based Fraud Detection
    df.groupby('transaction_date').agg({'is_fraud_predicted_rule': 'sum'}).plot(label='Rule-based Fraud', color='blue')
    # Plot for AI-based Fraud Detection
    df.groupby('transaction_date').agg({'is_fraud_predicted_ai': 'sum'}).plot(label='AI-based Fraud', color='red')
    
    # Format the x-axis to show time
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    
    plt.title('Time Series of Predicted Fraud Cases')
    plt.xlabel('Date')
    plt.ylabel('Number of Fraud Cases')
    plt.legend()
    plt.grid(True)

    # Add cursor to zoom
    cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=1)
    plt.show()

# Menu-driven program
def menu():
    while True:
        print("\nMenu:")
        print("1. View fraud detection status for all transactions.")
        print("2. Enter batch size and analyze specific transactions.")
        print("3. View time series graph of fraud predictions.")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ")
        
        if choice == '1':
            view_all_transactions_fraud_detection()
        elif choice == '2':
            batch_size = int(input("Enter the batch size (number of transaction IDs to check): "))
            evaluate_batch_transactions(batch_size)
        elif choice == '3':
            plot_time_series_graph()
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice! Please try again.")

# Run the menu-driven program
menu()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_FILE = "diabetes.csv"

def perform_eda():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run data_loader.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    print("\n--- Data Head ---")
    print(df.head())
    
    print("\n--- Data Info ---")
    print(df.info())
    
    print("\n--- Data Description ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Histograms
    plt.figure(figsize=(12, 10))
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.savefig("histograms.png")
    print("Saved histograms.png")
    plt.close() # Close the figure to free memory
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    print("Saved heatmap.png")
    plt.close()

if __name__ == "__main__":
    perform_eda()

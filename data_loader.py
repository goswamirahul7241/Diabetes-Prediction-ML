import pandas as pd
import requests
import os

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
OUTPUT_FILE = "diabetes.csv"
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

def download_data():
    print(f"Downloading data from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        # The raw data doesn't have a header, so we'll save it and then reload with headers
        with open("temp_diabetes.csv", "wb") as f:
            f.write(response.content)
            
        print("Download complete. Processing...")
        
        # Load and add headers
        df = pd.read_csv("temp_diabetes.csv", header=None, names=COLUMNS)
        
        # Save to final file
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Data saved to {OUTPUT_FILE}")
        print(df.head())
        
        # Clean up temp file
        os.remove("temp_diabetes.csv")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()

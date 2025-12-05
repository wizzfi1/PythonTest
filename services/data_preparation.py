import pandas as pd
import os
import zipfile
import re

class DataPreparation:

    def __init__(self, zip_path="data/dataset.zip", output_path="data/cleaned_products.csv"):
        self.zip_path = zip_path
        self.output_path = output_path

    def extract_zip(self):
        """Extract dataset.zip and return CSV path."""
        print("Extracting dataset.zip ...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")

        for file in os.listdir("data/"):
            if file.endswith(".csv"):
                print(f"Found dataset: data/{file}")
                return os.path.join("data/", file)

        raise FileNotFoundError("No CSV file found inside dataset.zip")

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        # Remove emojis and weird characters
        text = re.sub(r"[^\w\s\-.,]", "", str(text))
        # Convert to lowercase
        return text.lower().strip()

    def clean_dataset(self):
        csv_path = self.extract_zip()
        print("Loading dataset...")

        df = pd.read_csv(csv_path)
        print("Initial dataset shape:", df.shape)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Drop rows without description
        df.dropna(subset=["Description"], inplace=True)

        # Clean text-based columns
        df["Description"] = df["Description"].apply(self.clean_text)
        df["Country"] = df["Country"].apply(self.clean_text)
        df["StockCode"] = df["StockCode"].apply(self.clean_text)

        # Remove rows where Description became empty
        df = df[df["Description"] != ""]

        # Clean numeric columns
        def clean_numeric(x):
            x = re.sub(r"[^\d.-]", "", str(x))
            try:
                return float(x)
            except:
                return None

        df["Quantity"] = df["Quantity"].apply(clean_numeric)
        df["UnitPrice"] = df["UnitPrice"].apply(clean_numeric)
        df["CustomerID"] = df["CustomerID"].apply(clean_numeric)

        # Drop rows with invalid numeric values
        df.dropna(subset=["Quantity", "UnitPrice"], inplace=True)

        print("Cleaned dataset shape:", df.shape)

        # Save cleaned CSV
        df.to_csv(self.output_path, index=False)
        print(f"Cleaned dataset saved to: {self.output_path}")

        return self.output_path

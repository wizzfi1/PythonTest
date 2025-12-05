import pandas as pd
import re
import unicodedata

class ProductCatalogBuilder:

    def __init__(self, cleaned_csv_path="data/cleaned_products.csv"):
        self.cleaned_csv_path = cleaned_csv_path

    def make_ascii(self, text):
        if pd.isna(text):
            return ""
        # Normalize unicode → ASCII
        text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode()
        # Remove anything not alphanumeric
        text = re.sub(r'[^A-Za-z0-9]', '', text)
        return text

    def build_catalog(self, output_path="data/product_catalog.csv"):
        df = pd.read_csv(self.cleaned_csv_path)

        # Remove missing descriptions
        df = df[df["Description"].notna()]

        # CLEAN STOCKCODES TO ASCII
        df["StockCode"] = df["StockCode"].apply(self.make_ascii)

        # Drop empty IDs (after cleaning)
        df = df[df["StockCode"] != ""]

        # Group by cleaned StockCode
        grouped = df.groupby("StockCode").agg({
            "Description": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            "UnitPrice": "mean"
        }).reset_index()

        grouped.rename(columns={
            "StockCode": "product_id",
            "Description": "product_name",
            "UnitPrice": "price"
        }, inplace=True)

        grouped.to_csv(output_path, index=False)
        print(f"Product catalog saved to: {output_path}")
        print(f"Total products: {len(grouped)}")
        return output_path

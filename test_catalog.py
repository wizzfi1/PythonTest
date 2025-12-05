from services.product_catalog import ProductCatalogBuilder

builder = ProductCatalogBuilder("data/cleaned_products.csv")
builder.build_catalog()

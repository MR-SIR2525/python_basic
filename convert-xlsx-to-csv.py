import pandas as pd

# Load the Excel file
df = pd.read_excel('datasets/commerce_data.xlsx')

# Save as CSV
df.to_csv('datasets/sales_data.csv', index=False)

print("Conversion complete!")

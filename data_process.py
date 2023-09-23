import pandas as pd
import string
import re

# Load the CSV file into a DataFrame
file_path = 'outputsplit.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Define a function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation using regular expressions
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # You can add more preprocessing steps here if needed
    
    return text

# Apply the preprocessing function to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Save the preprocessed DataFrame to a new CSV file
output_file_path = 'preprocessed_data1.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)

print("Text preprocessing completed and saved to", output_file_path)

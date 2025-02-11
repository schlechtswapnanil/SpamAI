import pandas as pd

try:
    df = pd.read_csv('spam.csv', sep=',')  # Handle potential encoding issues
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    exit()
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Please check the file format.")
    exit()

# Check if 'label' and 'message' columns exist. If not, try other common names
if 'label' not in df.columns:
    if 'v1' in df.columns:
        df.rename(columns={'v1': 'label'}, inplace=True)
    elif 'Category' in df.columns:  # Check for 'Category' as a label column
        df.rename(columns={'Category': 'label'}, inplace=True)
    else:
        print("Error: 'label' column not found. Please check your CSV file.")
        exit()

if 'message' not in df.columns:
    if 'v2' in df.columns:
        df.rename(columns={'v2': 'message'}, inplace=True)
    elif 'Message' in df.columns: # Check for 'Message' as a message column
        df.rename(columns={'Message': 'message'}, inplace=True)
    elif 'text' in df.columns:
        df.rename(columns={'text': 'message'}, inplace=True)
    else:
        print("Error: 'message' column not found. Please check your CSV file.")
        exit()

#Handle potential NaN values in label column
df.dropna(subset=['label'], inplace=True)


# Calculate the total number of messages
total_messages = len(df)

# Calculate the number of spam messages
num_spam = len(df[df['label'].str.lower() == 'spam']) #Case insensitive comparison
num_ham = len(df[df['label'].str.lower() == 'ham']) #Case insensitive comparison

# Calculate the percentage of spam and ham messages
spam_percentage = (num_spam / total_messages) * 100
ham_percentage = (num_ham / total_messages) * 100

# Calculate message length statistics
df['message_length'] = df['message'].str.len()
avg_message_length = df['message_length'].mean()
min_message_length = df['message_length'].min()
max_message_length = df['message_length'].max()

# Print the results in a format suitable for LaTeX
print(f"\\textbf{{Number of messages:}} {total_messages}")
print(f"\\textbf{{Number of spam messages:}} {num_spam}")
print(f"\\textbf{{Number of ham messages:}} {num_ham}")
print(f"\\textbf{{Percentage of spam messages:}} {spam_percentage:.2f}\\%")
print(f"\\textbf{{Percentage of ham messages:}} {ham_percentage:.2f}\\%")
print(f"\\textbf{{Average message length:}} {avg_message_length:.2f}")
print(f"\\textbf{{Minimum message length:}} {min_message_length}")
print(f"\\textbf{{Maximum message length:}} {max_message_length}")

#Print column names and types for the data format section
print("\n\\textbf{{Column names and types:}}")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")
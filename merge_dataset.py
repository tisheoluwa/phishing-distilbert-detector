import pandas as pd

# Load both datasets
df_emails = pd.read_csv("data/phishing_email.csv")
df_urls = pd.read_csv("data/phishing_urls.csv")

# Combine them
df_combined = pd.concat([df_emails, df_urls], ignore_index=True)

# Optional: Shuffle the rows
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# Save the final dataset
df_combined.to_csv("data/phishing_mixed.csv", index=False)

print(f"✅ Dataset created: data/phishing_mixed.csv ({len(df_combined)} rows)")

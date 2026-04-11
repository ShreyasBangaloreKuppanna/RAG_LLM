import os
import pandas as pd
from thefuzz import process, fuzz

# 1. Define folder paths and subfolders
source_root = "data_2024_2/Bestandsdaten"
target_root = "data_2025"
subfolders = ["Hochspannung", "Mittelspannung", "Niederspannung"]
matching_score = 60
results = []

for folder in subfolders:
    src_path = os.path.join(source_root, folder)
    tgt_path = os.path.join(target_root, folder)

    # Get all files in this specific source subfolder
    src_files = [f for f in os.listdir(src_path) if f.endswith(('.xlsx', '.csv'))]
    # Get all files in the corresponding target subfolder
    tgt_files = [f for f in os.listdir(tgt_path) if f.endswith(('.xlsx', '.csv'))]

    for src_file in src_files:
        # Use TheFuzz to find the best match only within the SAME subfolder
        best_match, score = process.extractOne(
            src_file, tgt_files, scorer=fuzz.token_sort_ratio
        )
        # Save absolute paths for the LLM/Pandas to use later
        src_full_path = os.path.join(src_path, src_file)
        tgt_full_path = os.path.join(tgt_path, best_match) if score > matching_score else None

        match_entry = {
            "Category": folder,
            "Source File": src_file,
            "Source_Path": src_full_path,
            "Target Match": best_match if score > matching_score else "NOT FOUND",
            "Target_Path": tgt_full_path,
            "Fuzzy Score": score,
            "Status": "Matched" if score > matching_score else "Missing in 2025"
        }
        results.append(match_entry)

# 2. Save the Inventory Match Result
report_df = pd.DataFrame(results)
report_df.to_csv("inventory_alignment_report.csv", index=False)
print(report_df)
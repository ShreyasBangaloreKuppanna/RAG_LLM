import pandas as pd
import os
# from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOllama(model="llama3")

def find_id_column_mapping(df_src, df_tgt):
    """Ask LLM to find the ID columns based on samples."""
    # Get top 10 unique values from each column to keep prompt small
    src_samples = {col: df_src[col].dropna().unique()[:10].tolist() for col in df_src.columns}
    tgt_samples = {col: df_tgt[col].dropna().unique()[:10].tolist() for col in df_tgt.columns}

    prompt = f"""
    Find the primary ID/Key columns that match between these two tables.
    Source Columns & Samples: {src_samples}
    Target Columns & Samples: {tgt_samples}

    Return ONLY a JSON with keys 'source_col' and 'target_col'. 
    If no match is clear, return {{"source_col": null, "target_col": null}}.
    """

    response = llm.invoke(prompt)
    print('-----------------------------------------------------------', response)
    # Extract JSON from response (simplified for example)
    import json
    return json.loads(response.content.replace("```json", "").replace("```", ""))


# # Load your existing report
# report_df = pd.read_csv("inventory_alignment_report.csv")
#
# mappings = []
# for index, row in report_df.iterrows():
#     if row['Status'] == 'Matched':
#         # Load the actual matched files
#         df_s = pd.read_excel(row['Source_Path']) if row['Source_Path'].endswith('.xlsx') else pd.read_csv(row['Source_Path'])  #
#         # Paths should be in your report
#         df_t = pd.read_excel(row['Target_Path']) if row['Target_Path'].endswith('.xlsx') else pd.read_csv(row['Target_Path'])
#
#         mapping = find_id_column_mapping(df_s, df_t)
#         mappings.append(mapping)
#     else:
#         mappings.append({"source_col": None, "target_col": None})
#
# # Extend the report
# report_df['Source_ID_Col'] = [m['source_col'] for m in mappings]
# report_df['Target_ID_Col'] = [m['target_col'] for m in mappings]
# report_df.to_csv("final_inventory_mapping.csv", index=False)


source_df = pd.read_csv('data_2024_2/Bestandsdaten/Hochspannung/110kVLeistungsschalter.csv')
target_df = pd.read_csv('data_2025/Hochspannung/110kVLeistungsschalter.csv',  encoding = "ISO-8859-1" )
find_id_column_mapping(source_df, target_df)
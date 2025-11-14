# importing required libraries

import os
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
from tqdm import tqdm
import time

#reading files
train = pd.read_excel(r"\bodywash-train.xlsx")
test = pd.read_excel(r"C:\bodywash-test.xlsx")
# ----------------------------------
train = train[['Core Item', 'Level 1 (PARENT)', 'Level 2 (CHILD)']]
print(train.head(2))

train.rename(columns={'Level 1 (PARENT)':'Level 1',
                      'Level 2 (CHILD)': 'Level 2'}, inplace=True)

del test['Level 1']  
del test['Level 2']  
# --------------------------------
#Cleaning the text -- removing punctuations and making it lowercase
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Creating new column and storing so that the original text dosen't go.
train["New Core"] = train["Core Item"].apply(clean_text)
test["New Core"] = test["Core Item"].apply(clean_text)

#storing the tags of level 1 and level 2 in form of list
level1_tags = sorted(train["Level 1"].dropna().unique().tolist())
level2_tags = sorted(train["Level 2"].dropna().unique().tolist())

#using gemini 2.5 flash model
genai.configure(api_key="XXXYYYXXXYYYXXYYXXYXYXYX") # enter your keys
model = genai.GenerativeModel("gemini-2.5-flash")

# Giving the prompt to model
def make_prompt(text):
    return f"""
You are an expert text classifier.

Based on the review text, assign the most relevant Level 1 and Level 2 tags. 
Please remember that Level 2 is dependent on Level 1.

Possible Level 1 tags: {', '.join(level1_tags)}
Possible Level 2 tags: {', '.join(level2_tags)}

Output strictly in this format:
Level 1: [comma separated values]
Level 2: [comma separated values]

Text: "{text}"
"""
## Passing each and every 'Cleaned text and getting the l1 and l2'
preds_l1, preds_l2 = [], []
sleep_seconds = 3
max_retries = 3

for i, row in tqdm(test.iterrows(), total=len(test)):
    text = row["New Core"]
    prompt = make_prompt(text)
    l1, l2 = "", ""

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            reply = response.text.strip()

            # --- Flexible extraction patterns ---
            l1_match = re.search(r"Level\s*1[^:]*[:\-]\s*\[?(.*?)\]?(?:\n|$)", reply, re.IGNORECASE)
            l2_match = re.search(r"Level\s*2[^:]*[:\-]\s*\[?(.*?)\]?(?:\n|$)", reply, re.IGNORECASE)
            if not l1_match:
                l1_match = re.search(r"L1[:\-]\s*(.*)", reply)
            if not l2_match:
                l2_match = re.search(r"L2[:\-]\s*(.*)", reply)

            l1_raw = l1_match.group(1).strip() if l1_match else ""
            l2_raw = l2_match.group(1).strip() if l2_match else ""

            # Keep only the first tag
            l1 = l1_raw.split(",")[0].strip() if l1_raw else ""
            l2 = l2_raw.split(",")[0].strip() if l2_raw else ""

            # If Gemini still didn’t format correctly, try a fallback heuristic
            if not l1 or not l2:
                for tag in level1_tags:
                    if tag.lower() in reply.lower():
                        l1 = tag
                        break
                for tag in level2_tags:
                    if tag.lower() in reply.lower():
                        l2 = tag
                        break

            break

        except Exception as e:
            print(f"⚠️ Error at row {i}, attempt {attempt+1}: {e}")
            time.sleep(sleep_seconds * (attempt + 1))  # exponential backoff

    preds_l1.append(l1)
    preds_l2.append(l2)
    time.sleep(sleep_seconds)
   

# Combining predictions into DataFrame
test["Predicted Level 1"] = preds_l1
test["Predicted Level 2"] = preds_l2

#----------------------- Renaming it in original column 
test.rename(columns={'Predicted Level 1':'Level 1',
                     'Predicted Level 2': 'Level 2'}, inplace = True)
print(test.head(1))

test.to_excel(r"\bodywash-test-predicted.xlsx", index=False)


## For Accuracy : 

# Create Level 1 → Level 2 mapping for train and test
train_map = (
    train.groupby("Level 1")["Level 2"]
    .apply(lambda x: set(x.str.strip()))
    .to_dict()
)
test_map = (
    test.groupby("Level 1")["Level 2"]
    .apply(lambda x: set(x.str.strip()))
    .to_dict()
)

# Get only common Level 1 values
common_level1 = set(train_map.keys()).intersection(set(test_map.keys()))

results = []
for l1 in common_level1:
    train_tags = train_map[l1]
    test_tags = test_map[l1]

    common_tags = test_tags.intersection(train_tags)
    total_test = len(test_tags)
    total_common = len(common_tags)
    overlap_pct = round((total_common / total_test) * 100, 2) if total_test > 0 else 0

    results.append({
        "Level 1": l1,
        "Train_L2_Count": len(train_tags),
        "Test_L2_Count": total_test,
        "Common_L2_Count": total_common,
        "Common_L2_%": overlap_pct,
        "Common_L2_Tags": list(common_tags),
        "Missing_in_Test": list(train_tags - test_tags)
    })

# Convert to DataFrame
overlap_df = pd.DataFrame(results).sort_values(by="Common_L2_%", ascending=False)

print(overlap_df.head(3))


## Conculsion :
## 1  Accuracy approch:

"""Since test labels were unavailable, I evaluated how well my predicted tag distribution aligns with the training distribution by calculating the percentage of Level 2 tags under each Level 1 that overlap with the training data.
For common Level 1 tags, about 48% of the Level 2 predictions overlapped"""

## 2 Accuracy approch:

"""Since the test set didn’t have ground-truth tags, I performed a manual validation on a small sample of test reviews — checking whether the predicted Level 1 and Level 2 tags made sense contextually.
In my sample of 20 reviews, about 80–85% predictions were contextually correct, based on human judgment"""




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math


# In[2]:


# https://ray.so/#code=UFMgRDpcRXBoeXJ1c1xWUyBTb3VyY2UgQ29kZVw1MC4wMDdcMUQgUHJvamVjdFxFU1xFdmFsU2NyaXB0PiBweXRob24gZXZhbFJlc3VsdC5weSBkZXYub3V0IGRldi5wcmVkaWN0aW9uMSAgICAgCiNFbnRpdHkgaW4gZ29sZCBkYXRhOiAyMjkKI0VudGl0eSBpbiBwcmVkaWN0aW9uOiAxNDY2CiNDb3JyZWN0IEVudGl0eSA6IDE3OApFbnRpdHkgIHByZWNpc2lvbjogMC4xMjE0CkVudGl0eSAgcmVjYWxsOiAwLjc3NzMKRW50aXR5ICBGOiAwLjIxMDAKI0NvcnJlY3QgU2VudGltZW50IDogOTcKU2VudGltZW50ICBwcmVjaXNpb246IDAuMDY2MgpTZW50aW1lbnQgIHJlY2FsbDogMC40MjM2ClNlbnRpbWVudCAgRjogMC4xMTQ1CgoKClBTIEQ6XEVwaHlydXNcVlMgU291cmNlIENvZGVcNTAuMDA3XDFEIFByb2plY3RcRVNcRXZhbFNjcmlwdD4gcHl0aG9uIGV2YWxSZXN1bHQucHkgZGV2Lm91dCBkZXYucHJlZGljdGlvbjIgICAgIAojRW50aXR5IGluIGdvbGQgZGF0YTogMjI5CiNFbnRpdHkgaW4gcHJlZGljdGlvbjogMjU2CiNDb3JyZWN0IEVudGl0eSA6IDcyCkVudGl0eSAgcHJlY2lzaW9uOiAwLjI4MTIKRW50aXR5ICByZWNhbGw6IDAuMzE0NApFbnRpdHkgIEY6IDAuMjk2OQojQ29ycmVjdCBTZW50aW1lbnQgOiA2MQpTZW50aW1lbnQgIHByZWNpc2lvbjogMC4yMzgzClNlbnRpbWVudCAgcmVjYWxsOiAwLjI2NjQKU2VudGltZW50ICBGOiAwLjI1MTUKCgoKUFMgRDpcRXBoeXJ1c1xWUyBTb3VyY2UgQ29kZVw1MC4wMDdcMUQgUHJvamVjdFxFU1xFdmFsU2NyaXB0PiBweXRob24gZXZhbFJlc3VsdC5weSBkZXYub3V0IGRldi5wcmVkaWN0aW9uM19rXzEKI0VudGl0eSBpbiBnb2xkIGRhdGE6IDIyOQojRW50aXR5IGluIHByZWRpY3Rpb246IDQyNwojQ29ycmVjdCBFbnRpdHkgOiA3MApFbnRpdHkgIHByZWNpc2lvbjogMC4xNjM5CkVudGl0eSAgcmVjYWxsOiAwLjMwNTcKRW50aXR5ICBGOiAwLjIxMzQKI0NvcnJlY3QgU2VudGltZW50IDogNTYKU2VudGltZW50ICBwcmVjaXNpb246IDAuMTMxMQpTZW50aW1lbnQgIHJlY2FsbDogMC4yNDQ1ClNlbnRpbWVudCAgRjogMC4xNzA3CgoKClBTIEQ6XEVwaHlydXNcVlMgU291cmNlIENvZGVcNTAuMDA3XDFEIFByb2plY3RcRVNcRXZhbFNjcmlwdD4gcHl0aG9uIGV2YWxSZXN1bHQucHkgZGV2Lm91dCBkZXYucHJlZGljdGlvbjNfa18yIAojRW50aXR5IGluIGdvbGQgZGF0YTogMjI5CiNFbnRpdHkgaW4gcHJlZGljdGlvbjogNDg5CiNDb3JyZWN0IEVudGl0eSA6IDU2CkVudGl0eSAgcHJlY2lzaW9uOiAwLjExNDUKRW50aXR5ICByZWNhbGw6IDAuMjQ0NQpFbnRpdHkgIEY6IDAuMTU2MAojQ29ycmVjdCBTZW50aW1lbnQgOiAyNQpTZW50aW1lbnQgIHByZWNpc2lvbjogMC4wNTExClNlbnRpbWVudCAgcmVjYWxsOiAwLjEwOTIKU2VudGltZW50ICBGOiAwLjA2OTYKCgoKUFMgRDpcRXBoeXJ1c1xWUyBTb3VyY2UgQ29kZVw1MC4wMDdcMUQgUHJvamVjdFxFU1xFdmFsU2NyaXB0PiBweXRob24gZXZhbFJlc3VsdC5weSBkZXYub3V0IGRldi5wcmVkaWN0aW9uM19rXzMKI0VudGl0eSBpbiBnb2xkIGRhdGE6IDIyOQojRW50aXR5IGluIHByZWRpY3Rpb246IDQ4NgojQ29ycmVjdCBFbnRpdHkgOiA1MwpFbnRpdHkgIHByZWNpc2lvbjogMC4xMDkxCkVudGl0eSAgcmVjYWxsOiAwLjIzMTQKRW50aXR5ICBGOiAwLjE0ODMKI0NvcnJlY3QgU2VudGltZW50IDogMzMKU2VudGltZW50ICBwcmVjaXNpb246IDAuMDY3OQpTZW50aW1lbnQgIHJlY2FsbDogMC4xNDQxClNlbnRpbWVudCAgRjogMC4wOTIzCgoKClBTIEQ6XEVwaHlydXNcVlMgU291cmNlIENvZGVcNTAuMDA3XDFEIFByb2plY3RcRVNcRXZhbFNjcmlwdD4gcHl0aG9uIGV2YWxSZXN1bHQucHkgZGV2Lm91dCBkZXYucHJlZGljdGlvbjNfa180IAojRW50aXR5IGluIGdvbGQgZGF0YTogMjI5CiNFbnRpdHkgaW4gcHJlZGljdGlvbjogNTAzCiNDb3JyZWN0IEVudGl0eSA6IDQ4CkVudGl0eSAgcHJlY2lzaW9uOiAwLjA5NTQKRW50aXR5ICByZWNhbGw6IDAuMjA5NgpFbnRpdHkgIEY6IDAuMTMxMQojQ29ycmVjdCBTZW50aW1lbnQgOiAyNwpTZW50aW1lbnQgIHByZWNpc2lvbjogMC4wNTM3ClNlbnRpbWVudCAgcmVjYWxsOiAwLjExNzkKU2VudGltZW50ICBGOiAwLjA3MzgKCgoKUFMgRDpcRXBoeXJ1c1xWUyBTb3VyY2UgQ29kZVw1MC4wMDdcMUQgUHJvamVjdFxFU1xFdmFsU2NyaXB0PiBweXRob24gZXZhbFJlc3VsdC5weSBkZXYub3V0IGRldi5wcmVkaWN0aW9uM19rXzUgCiNFbnRpdHkgaW4gZ29sZCBkYXRhOiAyMjkKI0VudGl0eSBpbiBwcmVkaWN0aW9uOiA1MjUKI0NvcnJlY3QgRW50aXR5IDogNTIKRW50aXR5ICBwcmVjaXNpb246IDAuMDk5MApFbnRpdHkgIHJlY2FsbDogMC4yMjcxCkVudGl0eSAgRjogMC4xMzc5CiNDb3JyZWN0IFNlbnRpbWVudCA6IDI0ClNlbnRpbWVudCAgcHJlY2lzaW9uOiAwLjA0NTcKU2VudGltZW50ICByZWNhbGw6IDAuMTA0OApTZW50aW1lbnQgIEY6IDAuMDYzNwoKCgpQUyBEOlxFcGh5cnVzXFZTIFNvdXJjZSBDb2RlXDUwLjAwN1wxRCBQcm9qZWN0XEVTXEV2YWxTY3JpcHQ-IHB5dGhvbiBldmFsUmVzdWx0LnB5IGRldi5vdXQgZGV2LnByZWRpY3Rpb24zX2tfNgojRW50aXR5IGluIGdvbGQgZGF0YTogMjI5CiNFbnRpdHkgaW4gcHJlZGljdGlvbjogNTE3CiNDb3JyZWN0IEVudGl0eSA6IDQ1CkVudGl0eSAgcHJlY2lzaW9uOiAwLjA4NzAKRW50aXR5ICByZWNhbGw6IDAuMTk2NQpFbnRpdHkgIEY6IDAuMTIwNgojQ29ycmVjdCBTZW50aW1lbnQgOiAzMApTZW50aW1lbnQgIHByZWNpc2lvbjogMC4wNTgwClNlbnRpbWVudCAgcmVjYWxsOiAwLjEzMTAKU2VudGltZW50ICBGOiAwLjA4MDQKCgoKUFMgRDpcRXBoeXJ1c1xWUyBTb3VyY2UgQ29kZVw1MC4wMDdcMUQgUHJvamVjdFxFU1xFdmFsU2NyaXB0PiBweXRob24gZXZhbFJlc3VsdC5weSBkZXYub3V0IGRldi5wcmVkaWN0aW9uM19rXzcgCiNFbnRpdHkgaW4gZ29sZCBkYXRhOiAyMjkKI0VudGl0eSBpbiBwcmVkaWN0aW9uOiA1MTgKI0NvcnJlY3QgRW50aXR5IDogNTAKRW50aXR5ICBwcmVjaXNpb246IDAuMDk2NQpFbnRpdHkgIHJlY2FsbDogMC4yMTgzCkVudGl0eSAgRjogMC4xMzM5CiNDb3JyZWN0IFNlbnRpbWVudCA6IDIzClNlbnRpbWVudCAgcHJlY2lzaW9uOiAwLjA0NDQKU2VudGltZW50ICByZWNhbGw6IDAuMTAwNApTZW50aW1lbnQgIEY6IDAuMDYxNgoKCgpQUyBEOlxFcGh5cnVzXFZTIFNvdXJjZSBDb2RlXDUwLjAwN1wxRCBQcm9qZWN0XEVTXEV2YWxTY3JpcHQ-IHB5dGhvbiBldmFsUmVzdWx0LnB5IGRldi5vdXQgZGV2LnByZWRpY3Rpb24zX2tfOAojRW50aXR5IGluIGdvbGQgZGF0YTogMjI5CiNFbnRpdHkgaW4gcHJlZGljdGlvbjogNDk3CiNDb3JyZWN0IEVudGl0eSA6IDQxCkVudGl0eSAgcHJlY2lzaW9uOiAwLjA4MjUKRW50aXR5ICByZWNhbGw6IDAuMTc5MApFbnRpdHkgIEY6IDAuMTEyOQojQ29ycmVjdCBTZW50aW1lbnQgOiAxNApTZW50aW1lbnQgIHByZWNpc2lvbjogMC4wMjgyClNlbnRpbWVudCAgcmVjYWxsOiAwLjA2MTEKU2VudGltZW50ICBGOiAwLjAzODYKCgoKUFMgRDpcRXBoeXJ1c1xWUyBTb3VyY2UgQ29kZVw1MC4wMDdcMUQgUHJvamVjdFxFU1xFdmFsU2NyaXB0PiBweXRob24gZXZhbFJlc3VsdC5weSBkZXYub3V0IGRldi5wcmVkaWN0aW9uNCAgICAKI0VudGl0eSBpbiBnb2xkIGRhdGE6IDIyOQojRW50aXR5IGluIHByZWRpY3Rpb246IDIxMwojQ29ycmVjdCBFbnRpdHkgOiAxMzgKRW50aXR5ICBwcmVjaXNpb246IDAuNjQ3OQpFbnRpdHkgIHJlY2FsbDogMC42MDI2CkVudGl0eSAgRjogMC42MjQ0CiNDb3JyZWN0IFNlbnRpbWVudCA6IDEwOApTZW50aW1lbnQgIHByZWNpc2lvbjogMC41MDcwClNlbnRpbWVudCAgcmVjYWxsOiAwLjQ3MTYKU2VudGltZW50ICBGOiAwLjQ4ODc&padding=32&theme=midnight&background=true&language=shell&title=ES+Dataset


# # Part 1

# $p\left(x_1, \ldots, x_n, y_1, \ldots, y_n\right)=\prod_{i=1}^{n+1} q\left(y_i \mid y_{i-1}\right) \cdot \prod_{i=1}^n e\left(x_i \mid y_i\right)$

# In[3]:


# Read the training data from 'train.txt' file.
df = pd.read_csv(
    "train.txt",
    sep=" ",
    error_bad_lines=False,
    engine="python",
    quoting=3,
    dtype=str,
    encoding="utf-8",
    names=["Token", "Label"],
)


# In[4]:


display(df)


# In[5]:


# Extract the 'Token' and 'Label' columns separately.
df_token = df.loc[:, ["Token"]]
df_label = df.loc[:, ["Label"]]

# Convert the 'Token' and 'Label' dataframes to numpy arrays for efficient operations.
arr_token = (df_token.squeeze()).to_numpy()
arr_label = (df_label.squeeze()).to_numpy()

# Get the unique tokens and labels from the dataset.
token_all = df_token.squeeze().unique()
label_all = df_label.squeeze().unique()


# In[6]:


print(label_all)


# In[7]:


# Construct a dataframe for labels with their counts and locations in the original dataset.
y_all = pd.DataFrame(label_all, columns=["Label"])
y_all["Location"] = [np.where(arr_label == i)[0] for i in label_all]
y_all["Count"] = [len(y_all.loc[i, "Location"]) for i in range(len(y_all))]

print(len(y_all))
print(type(y_all))
display(y_all)


# In[8]:


# Construct a dataframe for tokens.
x_all = pd.DataFrame(token_all, columns=["Token"])

# This takes >1m to do.
# Expand the dataframe by repeating each token 7 times (assuming 7 is a predetermined number based on the dataset).
new_x_all = pd.DataFrame({"Token": np.repeat(x_all["Token"], 7)}).reset_index()
new_x_all.columns = ["Index", "Token"]

# Associate each repeated token with a label.
new_x_all["Label"] = np.tile(label_all, int(len(new_x_all) / 7))

# For each token-label pair, count the occurrences and note the locations in the original dataset.
count_x_all = []
location_x_all = []

for i in token_all:
    for j in label_all:
        cond_1 = arr_token == i
        cond_2 = arr_label == j
        loc = (np.where(cond_1 & cond_2)[0]).tolist()
        count_x_all += [len(loc)]
        location_x_all += [loc]

# Append the counts and locations to the expanded dataframe.
new_x_all["Count"] = count_x_all
new_x_all["Location"] = location_x_all

print(len(new_x_all))
print(type(new_x_all))
display(new_x_all)


# $1. \:\:\: e(x \mid y)=\frac{\operatorname{Count}(y \rightarrow x)}{\operatorname{Count}(y)}$

# In[9]:


def emm_par(x, y):
    """
    Calculate the emission parameter for a given token-label pair.

    Parameters:
    - x (str): An instance of Token.
    - y (str): An instance of Label.

    Returns:
    - float: Emission parameter value for the token-label pair.
    """

    # Define condition to filter rows in 'new_x_all' dataframe where 'Token' matches x.
    cond_1 = new_x_all["Token"] == x
    # Define condition to filter rows in 'new_x_all' dataframe where 'Label' matches y.
    cond_2 = new_x_all["Label"] == y
    # Extract the count value for the specific token-label combination.
    num = ((new_x_all[cond_1 & cond_2])["Count"]).values[0]

    # Define condition to filter rows in 'y_all' dataframe where 'Label' matches y.
    cond_3 = y_all["Label"] == y
    # Extract the total count value for the specific label.
    denom = ((y_all[cond_3])["Count"]).values[0]

    # Return the emission parameter, calculated as the count of the token-label pair divided by the total count of the label.
    return num / denom


# $2. \:\:\:e(x \mid y)=\left\{\begin{array}{cc}
# \frac{\operatorname{count}(y \rightarrow x)}{\operatorname{Count}(y)+k} & \text { If the word token } x \text { appears in the training set } \\
# \frac{k}{\operatorname{Count}(y)+k} & \text { If word token } x \text { is the special token \#UNK\# }
# \end{array}\right.$

# In[10]:


def emm_par_test(x, y):
    """
    Calculate the smoothed emission parameter for a given token-label pair, using a simple add-k smoothing.

    Parameters:
    - x (str): An instance of Token.
    - y (str): An instance of Label.

    Returns:
    - float: Smoothed emission parameter value for the token-label pair.
    """

    # Initialize the smoothing constant 'k'.
    k = 1

    # If the token 'x' exists in the 'new_x_all' dataframe, retrieve its count for the given label 'y'.
    if x in new_x_all.loc[:, "Token"].unique():
        cond_1 = new_x_all["Token"] == x
        cond_2 = new_x_all["Label"] == y
        num = ((new_x_all[cond_1 & cond_2])["Count"]).values[0]
    else:
        # If the token 'x' does not exist, assign the count as the smoothing constant 'k'.
        num = k

    # Retrieve the total count for the given label 'y'.
    cond_3 = y_all["Label"] == y
    denom = ((y_all[cond_3])["Count"]).values[0] + k  # Add 'k' for smoothing.

    # Return the smoothed emission parameter, calculated as the count of the token-label pair divided by the smoothed total count of the label.
    return num / denom


# In[11]:


# Initialize an empty list to store the emission parameter scores for each token-label pair.
emm_par_score_all = []
# Add a special token "#UNK#" (denoting 'unknown') to the list of all tokens.
new_token_all = np.append(token_all, "#UNK#")

print(new_token_all)


# In[12]:


# Need to check more, this takes >2m to do
# Iterate through each token in the dataset.
for i in token_all:
    # Initialize a variable to accumulate the emission parameter score for the current token across all labels.
    emm_par_score = 0
    # For each label, compute the emission parameter score using the 'emm_par_test' function.
    for j in label_all:
        emm_par_score = emm_par_test(i, j)
        # Append the computed score to the 'emm_par_score_all' list.
        emm_par_score_all += [emm_par_score]


# In[13]:


# Add the computed emission parameter scores as a new column "e(x|y)" in the 'new_x_all' dataframe.
new_x_all["e(x|y)"] = emm_par_score_all
display(new_x_all)


# $3. \:\:\:y^*=\underset{y}{\arg \max } \:e(x \mid y)$

# In[14]:


# Initialize an empty DataFrame for future use.
df_y = pd.DataFrame()

# Initialize an empty list to store the emission parameter scores for each token.
emm_par_score_all = []
# Initialize an empty list to store the most likely label (having the highest emission score) for each token.
y_star_all = []
# Add a special token "#UNK#" (denoting 'unknown') to the list of all tokens.
new_token_all = np.append(token_all, "#UNK#")

print(new_token_all)
print(len(new_token_all))


# In[15]:


# Need to check more, this takes >2m to do
# Iterate through each token in the updated token list.
for i in new_token_all:
    # Initialize variables to keep track of the highest emission score and the corresponding label for the current token.
    emm_par_score = 0
    y_star = ""

    # For each label, compute the emission parameter score using the 'emm_par_test' function.
    for j in label_all:
        emm_par_score_temp = emm_par_test(i, j)

        # If the current emission score is greater than the previously stored highest score, update the score and the corresponding label.
        if emm_par_score_temp > emm_par_score:
            emm_par_score = emm_par_score_temp
            y_star = j
        else:
            pass
    # Append the most likely label and its emission score for the current token to their respective lists.
    y_star_all += [y_star]
    emm_par_score_all += [emm_par_score]


# In[16]:


print(y_star_all)
print(len(y_star_all))
print(emm_par_score_all)
print(len(emm_par_score_all))


# In[17]:


# Add the 'new_token_all' list as a new column "Token" in the 'df_y' DataFrame.
df_y["Token"] = new_token_all
# Add the 'y_star_all' list (most likely label for each token) as a new column "y*" in the 'df_y' DataFrame.
df_y["y*"] = y_star_all
# Add the 'emm_par_score_all' list (highest emission score for each token) as a new column "Max e(x|y)" in the 'df_y' DataFrame.
df_y["Max e(x|y)"] = emm_par_score_all

display(df_y)


# In[18]:


# Convert the 'df_y' DataFrame to a dictionary where each key is a token, and the corresponding value is a tuple containing the most likely label and its emission score.
df_y_dict = {
    row["Token"]: (row["y*"], row["Max e(x|y)"])
    for row in df_y.to_dict(orient="records")
}
print(df_y_dict)


# ### Emission Probability Testing

# In[19]:


# df_test = pd.read_csv('dev_out.txt', sep=' ', error_bad_lines=False, engine="python", quoting=3,  dtype=str, encoding='utf-8', names=["Token", "Label"])

# Open the file 'dev_out.txt' and read its content.
with open("dev_out.txt", "r") as f:
    lines = f.read().splitlines()

# For each line in the file, split the line into two parts (Token and Label).
# If a line is empty, replace it with two empty strings ["", ""].
data = [line.split(" ") if line else ["", ""] for line in lines]

# Convert the processed data (list of lists) into a pandas DataFrame.
df_test_original = pd.DataFrame(data, columns=["Token", "Label"])


# In[20]:


# Create a copy of the original DataFrame to work with.
df_test = df_test_original.copy()

# Extract the 'Token' and 'Label' columns separately.
df_test_token = df_test.loc[:, ["Token"]]
df_test_label = df_test.loc[:, ["Label"]]

# Convert the extracted columns to numpy arrays for further processing.
arr_test_token = (df_test_token.squeeze()).to_numpy()
arr_test_label = (df_test_label.squeeze()).to_numpy()

# Get the unique tokens and labels from the test dataset.
token_test_all = df_test_token.squeeze().unique()
label_test_all = df_test_label.squeeze().unique()


# In[21]:


# Initialize an empty list to store predicted labels.
test_pred = []
# Initialize an empty list to store new tokens (those not found in training data).
new_token = []

# For each token in the test dataset, predict its label using the pre-computed dictionary 'df_y_dict'.
for i in arr_test_token:
    try:
        # Try to get the predicted label for the token from the dictionary.
        pred = df_y_dict[i][0]
        test_pred += [pred]
    except:
        # If the token is not found in the dictionary or is an empty string, handle the exception.
        if not i:
            pred = ""
        else:
            # For unknown tokens, use the label associated with the special token "#UNK#".
            pred = df_y_dict["#UNK#"][0]
        # Add the predicted label to the list.
        test_pred += [pred]
        # If the token is a new (unseen) token, add it to the 'new_token' list.
        new_token += [i]


# In[22]:


# Add the list of predicted labels as a new column to the 'df_test' DataFrame.
df_test["Predicted Label (Part 1)"] = test_pred
display(df_test)

# Convert the predicted labels column back to a numpy array for potential further processing.
arr_test_pred_label = (
    (df_test.loc[:, ["Predicted Label (Part 1)"]]).squeeze()
).to_numpy()


# In[23]:


df_out = df_test.loc[:, ["Token", "Predicted Label (Part 1)"]]
# df_out.to_csv('EvalScript/dev.prediction1', sep=' ', index=False, header=False)
df_out.to_csv("dev.p1.out", sep=" ", index=False, header=False)


# # Part 2

# In[24]:


# df_test = pd.read_csv('dev_out.txt', sep=' ', error_bad_lines=False, engine="python", quoting=3,  dtype=str, encoding='utf-8', names=["Token", "Label"])

# Open the file 'train.txt' and read its content.
with open("train.txt", "r") as f:
    lines = f.read().splitlines()

# For each line in the file, split the line into two parts (Token and Label).
# If a line is empty, replace it with two empty strings ["", ""], indicating no token or label.
data = [line.split(" ") if line else ["", ""] for line in lines]

# Convert the processed data (list of lists) into a pandas DataFrame.
df_ss = pd.DataFrame(data, columns=["Token", "Label (yi-1)"])


# In[25]:


display(df_ss)


# In[26]:


# Extract the 'Label (yi-1)' column for further processing.
df_label_ss = df_ss.loc[:, ["Label (yi-1)"]]
# Convert the 'Label (yi-1)' column to a numpy array for easier handling in subsequent operations.
arr_label_ss = (df_label_ss.squeeze()).to_numpy()


# In[27]:


# Insert the special label 'START' at the beginning and 'STOP' at the end of the label array.
label_all_ss = np.insert(label_all, 0, "START")
label_all_ss = np.append(label_all_ss, "STOP")


# In[28]:


# Convert the modified label array to a pandas DataFrame.
y_all_ss = pd.DataFrame(label_all_ss, columns=["Label (yi-1)"])


# In[29]:


display(df_label_ss)
display(arr_label_ss)
display(label_all_ss)
display(y_all_ss)


# In[30]:


# Create a copy of the DataFrame with column name changed to 'Label' for subsequent operations.
new_x_all_ss = y_all_ss.copy()
new_x_all_ss.columns = ["Label"]
display(new_x_all_ss)


# In[31]:


# Create a DataFrame that repeats each 'Label (yi-1)' 9 times, to account for all possible destination labels (yi).
new_y_all_ss = pd.DataFrame(
    {"Label (yi-1)": np.repeat(y_all_ss["Label (yi-1)"], 9)}
).reset_index()
new_y_all_ss.columns = ["Index", "Label (yi-1)"]

# Tile the labels so that each source label ('Label (yi-1)') is paired with every possible destination label ('Destination Label (yi)').
new_y_all_ss["Destination Label (yi)"] = np.tile(
    label_all_ss, int(len(new_y_all_ss) / 9)
)

# Initialize a count column to store the number of occurrences of each (yi-1, yi) pair.
new_y_all_ss["Count (yi-1,yi)"] = np.zeros(len(new_y_all_ss), dtype=int)

# Initialize a label to represent the previous label in the sequence.
previous_y = "START"

# Iterate through the list of labels, updating the count for each observed (yi-1, yi) pair.
for i in arr_label_ss:
    if i == "":
        i = "STOP"
    current_y = i
    # Increment the count for the observed (yi-1, yi) pair.
    new_y_all_ss.loc[
        (new_y_all_ss["Label (yi-1)"] == previous_y)
        & (new_y_all_ss["Destination Label (yi)"] == current_y),
        "Count (yi-1,yi)",
    ] += 1
    if current_y == "STOP":
        current_y = "START"
    previous_y = current_y

# Display the updated DataFrame with counts for each (yi-1, yi) pair.
display(new_y_all_ss)


# In[32]:


# Calculate the total counts for each 'Label (yi-1)' over all its destination labels 'Label (yi)'.
# For each 'Label (yi-1)', sum the counts over the next 9 rows (representing all possible 'Label (yi)' values).
y_all_ss_sums = [
    int(new_y_all_ss.iloc[i : i + 9]["Count (yi-1,yi)"].sum())
    for i in range(0, len(new_y_all_ss), 9)
]
print(y_all_ss_sums)


# Update the 'Count' column in 'new_x_all_ss' DataFrame with the computed sums for each 'Label (yi-1)'.
new_x_all_ss["Count"] = y_all_ss_sums
# Set the count for the 'STOP' label to be the same as the count for the 'START' label.
new_x_all_ss.loc[new_x_all_ss["Label"] == "STOP", "Count"] = new_x_all_ss.loc[
    new_x_all_ss["Label"] == "START", "Count"
].values[0]
# Display the updated 'new_x_all_ss' DataFrame.
display(new_x_all_ss)


# $2. \:\:\:q\left(y_i \mid y_{i-1}\right)=\frac{\operatorname{Count}\left(y_{i-1}, y_i\right)}{\operatorname{Count}\left(y_{i-1}\right)}$

# In[33]:


# Define a function to calculate the transition probability q(yi|yi-1) for given labels yi and yi-1.
def q_test(x, y):
    # Condition to filter rows based on the 'Destination Label (yi)' value.
    cond_1 = new_y_all_ss["Label (yi-1)"] == y
    # Condition to filter rows based on the 'Label (yi-1)' value.
    cond_2 = new_y_all_ss["Destination Label (yi)"] == x
    # Extract the count of transitions from 'Label (yi-1)' to 'Destination Label (yi)'.
    num = ((new_y_all_ss[cond_1 & cond_2])["Count (yi-1,yi)"]).values[0]
    # Condition to filter rows based on the 'Label' value in new_x_all_ss DataFrame.
    cond_3 = new_x_all_ss["Label"] == y
    # Extract the total count of 'Label (yi-1)' occurrences.
    denom = ((new_x_all_ss[cond_3])["Count"]).values[0]
    # If the denominator is zero, return 0 to avoid division by zero.
    if not denom:
        return 0
    # Return the calculated transition probability.
    return num / denom


# In[34]:


# List to store all computed q-values.
q_value_all = []

# Calculate q-values for all combinations of labels yi and yi-1.
for i in label_all_ss:
    for j in label_all_ss:
        q_value = q_test(j, i)
        q_value_all += [q_value]
print(q_value_all)


# In[35]:


# Add the computed q-values as a new column 'q(yi|yi-1)' in the new_y_all_ss DataFrame.
new_y_all_ss["q(yi|yi-1)"] = q_value_all


# In[36]:


display(new_y_all_ss)


# In[37]:


display(new_x_all)


# $3. \:\:\:y_1^*, \ldots, y_n^*=\underset{y_1, \ldots, y_n}{\arg \max } \:\:p\left(x_1, \ldots, x_n, y_1, \ldots, y_n\right)$

# In[38]:


import numpy as np


def viterbi_algorithm(sent, transition_probs, emission_probs, tags, all_states):
    # Create a matrix to store the maximum probabilities for each tag at each position in the sentence.
    viterbi = np.zeros((len(tags), len(sent)))
    # Create a matrix to store the tag (as an index) that leads to the maximum probability at each position.
    backpointer = np.zeros((len(tags), len(sent)), "int")

    # Initialization step: Set up the starting probabilities for the first word in the sentence.
    for i, tag in enumerate(tags):
        # Handle cases where the first word is not in the training data.
        if sent[0] not in all_states:
            viterbi[i, 0] = transition_probs[("START", "I-neutral")]
        # If the tag and word combination exists in the emission probabilities, use it.
        elif (tag, sent[0]) in emission_probs:
            viterbi[i, 0] = (
                transition_probs[("START", tag)] * emission_probs[(tag, sent[0])]
            )
        else:
            # Assign a small probability for unknown words.
            viterbi[i, 0] = transition_probs[("START", tag)] * 1e-10

    # Recursion step: Move through the sentence and compute the best tag sequence.
    for t in range(1, len(sent)):
        for i, tag in enumerate(tags):
            # Handle words not in the training data.
            if sent[t] not in all_states:
                max_prob, best_tag = (
                    viterbi[i, t - 1] * transition_probs[(tags[i], "I-neutral")],
                    i,
                )
            else:
                # Compute the maximum probability considering all possible previous tags.
                max_prob, best_tag = max(
                    (
                        viterbi[j, t - 1]
                        * transition_probs[(tags[j], tag)]
                        * emission_probs.get((tag, sent[t]), 1e-10),
                        j,
                    )
                    for j in range(len(tags))
                )
            # Store the maximum probability and the tag that led to it.
            viterbi[i, t] = max_prob
            backpointer[i, t] = best_tag

    # Termination step: Determine the best ending tag and its probability.
    best_path_pointer = np.argmax(
        [viterbi[i, -1] * transition_probs[(tags[i], "STOP")] for i in range(len(tags))]
    )
    best_path_prob = (
        viterbi[best_path_pointer, -1]
        * transition_probs[(tags[best_path_pointer], "STOP")]
    )

    # Backtrack to determine the best path through the tags for the entire sentence.
    best_path = []
    for t in range(len(sent) - 1, -1, -1):
        best_path.insert(0, tags[best_path_pointer])
        best_path_pointer = backpointer[best_path_pointer, t]

    # Return the best tag sequence and its probability.
    return best_path, best_path_prob


# ### Viterbi Algorithm Testing

# In[39]:


# Define a function to create starting probabilities for each label.
def create_start_prob(df1, df2):
    # Get the count of transitions from 'START' label.
    count_values = df1.loc[df1["Label (yi-1)"] == "START", "Count (yi-1,yi)"].tolist()
    # Get the total count of 'START' label occurrences.
    count_start = df2.loc[df2["Label"] == "START", "Count"].values[0]
    # Calculate the probability for each label after 'START'.
    list1 = [value / count_start for value in count_values]
    list2 = label_all_ss.tolist()
    return dict(zip(list2, list1))


# Define a function to create transition probabilities between labels.
def create_trans_prob(df):
    trans_prob = {}
    for index, row in df.iterrows():
        yi_1 = row["Label (yi-1)"]
        yi = row["Destination Label (yi)"]
        prob = row["q(yi|yi-1)"]
        if yi_1 not in trans_prob:
            trans_prob[yi_1] = {}
        trans_prob[yi_1][yi] = prob
    return trans_prob


# Define a function to create emission probabilities for each token-label pair.
def create_emit_prob(df):
    emit_prob = {}
    for index, row in df.iterrows():
        label = row["Label"]
        token = row["Token"]
        prob = row["e(x|y)"]
        if label not in emit_prob:
            emit_prob[label] = {}
        emit_prob[label][token] = prob
    return emit_prob


# Define a function to split the observed sequences based on the empty tokens.
def create_observed_states(df):
    big_list = []
    small_list = []
    for token in df["Token"]:
        if token.strip() != "":
            small_list.append(token)
        else:
            big_list.append(small_list)
            small_list = []
    return big_list


# Define a function to transform a nested dictionary into a flat dictionary.
def transform_dict(old_dict):
    new_dict = {}
    for key_outer in old_dict.keys():
        for key_inner in old_dict[key_outer].keys():
            new_key = (key_outer, key_inner)
            new_dict[new_key] = old_dict[key_outer][key_inner]
    return new_dict


# Define a function to create emission probabilities in a different format.
def create_emit_probs(df):
    emit_prob = {}
    for index, row in df.iterrows():
        label = row["Label"]
        token = row["Token"]
        prob = row["e(x|y)"]
        key = (label, token)
        emit_prob[key] = prob
    return emit_prob


# In[40]:


# Extract all unique states (tokens) from the data.
all_states = token_all.tolist()
print(all_states)

# Create observed sequences based on the test data.
sent = create_observed_states(df_test)
print(sent)

# List of possible tags (labels).
tags = [
    "O",
    "B-positive",
    "B-negative",
    "B-neutral",
    "I-neutral",
    "I-positive",
    "I-negative",
]
print(tags)

# Compute and display the starting probabilities.
start_prob = create_start_prob(new_y_all_ss, new_x_all_ss)
print(start_prob)

# Compute and display the transition probabilities.
trans_prob = create_trans_prob(new_y_all_ss)
print(trans_prob)

# Transform and display the transition probabilities.
trans_probs = transform_dict(trans_prob)
print(trans_probs)

# Compute and display the emission probabilities.
emit_prob = create_emit_prob(new_x_all)
print(emit_prob)

# Compute and display the emission probabilities in a different format.
emit_probs = create_emit_probs(new_x_all)
print(emit_probs)


# In[41]:


# Initialize empty lists to store the best paths and their corresponding probabilities.
all_best_path = []
all_best_path_prob = []

# Loop through each observed sequence in the test data.
for i in sent:
    # Compute the best path and its probability for the current observed sequence using the Viterbi algorithm.
    best_path, best_path_prob = viterbi_algorithm(
        i, trans_probs, emit_probs, tags, all_states
    )
    # Append an empty string to the end of the best path to align with the original data format.
    best_path += [""]
    # Store the best path and its probability in their respective lists.
    all_best_path += best_path
    all_best_path_prob += [best_path_prob]

# Display the computed best paths and their probabilities.
print(all_best_path)
print(all_best_path_prob)


# In[42]:


# Add the computed best paths as a new column to the test DataFrame.
df_test["Predicted Label (Part 2)"] = all_best_path
display(df_test)


# In[43]:


df_out = df_test.loc[:, ["Token", "Predicted Label (Part 2)"]]
# df_out.to_csv('EvalScript/dev.prediction2', sep=' ', index=False, header=False)
df_out.to_csv("dev.p2.out", sep=" ", index=False, header=False)


# ## Part 3

# In[44]:


import heapq


def k_best_viterbi(
    observations,
    transition_probability,
    emission_probability,
    start_probability,
    states,
    all_states,
    k=1,
):
    """
    Implements the K-best Viterbi algorithm to determine the K most probable sequences of hidden states.

    Parameters:
    - observations: A sequence of observations.
    - states: A set of possible hidden states.
    - start_probability: A dictionary specifying the starting probabilities for each state.
    - transition_probability: A dictionary specifying the transition probabilities between states.
    - emission_probability: A dictionary specifying the probabilities of observations given states.
    - k: The number of sequences to return.
    - all_states: A set of all words that have been trained on.

    Returns:
    - A list of the k most probable sequences with their probabilities.
    """

    # Initialize the tables to store the K-best probabilities (V) and paths (path) for each state at each observation.
    V = [{} for _ in range(len(observations))]
    path = [{} for _ in range(len(observations))]

    # Initialization step: compute probabilities for the first observation.
    for state in states:
        # Calculate the emission probability for the current observation and state.
        obs_prob = (
            emission_probability[state][observations[0]]
            if observations[0] in all_states
            else 0
        )
        V[0][state] = [start_probability[state] * obs_prob]
        path[0][state] = [[state if obs_prob > 0 else "I-neutral"]]

    # Recursion step: compute the K-best probabilities and paths for each subsequent observation.
    for t in range(1, len(observations)):
        for current_state in states:
            # Handle out-of-vocabulary observations.
            if observations[t] not in all_states:
                V[t][current_state] = [0] * k
                path[t][current_state] = [["I-neutral"] * (t + 1)] * k
                continue

            # For each state, compute the probabilities and paths leading up to it from all possible previous states.
            all_probs_paths = []
            for previous_state in states:
                for i in range(k):
                    if (
                        t - 1 < len(V)
                        and previous_state in V[t - 1]
                        and i < len(V[t - 1][previous_state])
                    ):
                        prob = (
                            V[t - 1][previous_state][i]
                            * transition_probability[previous_state][current_state]
                            * emission_probability[current_state][observations[t]]
                        )
                        current_path = path[t - 1][previous_state][i] + [current_state]
                        all_probs_paths.append((prob, current_path))

            # Sort by probabilities and keep only the K-best for the current state.
            all_probs_paths.sort(reverse=True)
            V[t][current_state] = [item[0] for item in all_probs_paths[:k]]
            path[t][current_state] = [item[1] for item in all_probs_paths[:k]]

    # Termination step: gather the K-best paths and their probabilities for the last observation.
    k_best_paths = []
    for state in states:
        for i in range(k):
            if state in V[-1] and i < len(V[-1][state]):
                heapq.heappush(k_best_paths, (-V[-1][state][i], path[-1][state][i]))

    # Extract the top K sequences from the priority queue.
    k_best_sequences = []
    for _ in range(k):
        if k_best_paths:
            prob, seq = heapq.heappop(k_best_paths)
            k_best_sequences.append((seq, -prob))

    return k_best_sequences


# ### Kth Best Viterbi Algorithm Testing (K=2)

# In[45]:


# Set the number of best paths to be found.
k = 2
# Initialize a dictionary to store k best paths.
# Each key corresponds to a rank (1st best, 2nd best, etc.) and its value is initialized with an empty string.
all_best_paths = {key: [""] for key in range(1, k + 1)}

# Loop through each observed sequence in the test data.
for i in sent:
    # Compute the k best paths for the current observed sequence using the k-best Viterbi algorithm.
    best_paths = k_best_viterbi(
        i, trans_prob, emit_prob, start_prob, tags, all_states, k
    )
    # Store each of the k best paths in the 'all_best_paths' dictionary.
    for j in range(k):
        best_path = best_paths[j][0]
        # Append an empty string to the end of each best path to align with the original data format.
        best_path += [""]
        # Update the dictionary with the computed best path.
        all_best_paths[j + 1] += best_path


# In[46]:


# Create a copy of the original test DataFrame.
df_test_2 = df_test_original.copy()

# Add the k best paths as new columns to the test DataFrame.
for i in range(k):
    col = "Predicted Label (Part 3) K=" + str(i + 1)
    label = all_best_paths[i + 1]
    # Remove the initial empty string added during the dictionary initialization.
    label = label[1:]
    # Assign the computed labels to the new column.
    df_test_2[col] = label

# Display the updated test DataFrame with the k best paths.
display(df_test_2)


# In[47]:


# Iterate through the range of k (number of best paths).
for i in range(k):
    # Construct the column name based on the current iteration (k value).
    col = "Predicted Label (Part 3) K=" + str(i + 1)
    # Extract the relevant columns ("Token" and the predicted label column for the current k) from the dataframe.
    df_out = df_test_2.loc[:, ["Token", col]]
    # Construct the output file path using the current k value.
    loc = "EvalScript/dev.prediction3_k_" + str(i + 1)
    # Save the extracted data to a CSV file with the specified path, without an index and without headers.
    df_out.to_csv(loc, sep=" ", index=False, header=False)


# In[48]:


df_out = df_test_2.loc[
    :, ["Token", "Predicted Label (Part 3) K=1", "Predicted Label (Part 3) K=2"]
]
df_out.to_csv("dev.p3.2nd.out", sep=" ", index=False, header=False)


# ### Kth Best Viterbi Algorithm Testing (K=8)

# In[49]:


# Set the value of k (number of best paths) to 8.
k = 8
# Initialize a dictionary to store the best paths for each value from 1 to k.
all_best_paths = {key: [""] for key in range(1, k + 1)}

# Iterate over each sentence in the 'sent' list.
for i in sent:
    # Get the k best paths using the Viterbi algorithm for the current sentence.
    best_paths = k_best_viterbi(
        i, trans_prob, emit_prob, start_prob, tags, all_states, k
    )
    # For each path in the range of k...
    for j in range(k):
        try:
            # Try to get the best path from the best_paths list.
            best_path = best_paths[j][0]
        except:
            # If there's an error (e.g., index out of range), use the previous best path.
            best_path = best_paths[j - 1][0]
        # Append the best path to the all_best_paths dictionary.
        all_best_paths[j + 1] += best_path
    # Add empty strings to all paths in the all_best_paths dictionary.
    for l in range(k):
        all_best_paths[l + 1] += [""]


# In[50]:


# Create a copy of the original test dataframe.
df_test_8 = df_test_original.copy()

# For each value in the range of k...
for i in range(k):
    # Construct the column name based on the current k value.
    col = "Predicted Label (Part 3) K=" + str(i + 1)
    # Get the label predictions for the current k from the all_best_paths dictionary.
    label = all_best_paths[i + 1]
    # Adjust for the initial empty string.
    label = label[1:]
    # Assign the label predictions to the dataframe.
    df_test_8[col] = label

# Display the modified dataframe with predictions.
display(df_test_8)


# In[51]:


# For each value in the range of k...
for i in range(k):
    # Construct the column name based on the current k value.
    col = "Predicted Label (Part 3) K=" + str(i + 1)
    # Extract the relevant columns ("Token" and the current k's predicted label column) from the dataframe.
    df_out = df_test_8.loc[:, ["Token", col]]
    # Construct the output file path using the current k value.
    loc = "EvalScript/dev.prediction3_k_" + str(i + 1)
    # Save the extracted data to a CSV file with the specified path, without an index and without headers.
    df_out.to_csv(loc, sep=" ", index=False, header=False)


# In[52]:


df_out = df_test_8.loc[
    :,
    [
        "Token",
        "Predicted Label (Part 3) K=1",
        "Predicted Label (Part 3) K=2",
        "Predicted Label (Part 3) K=3",
        "Predicted Label (Part 3) K=4",
        "Predicted Label (Part 3) K=5",
        "Predicted Label (Part 3) K=6",
        "Predicted Label (Part 3) K=7",
        "Predicted Label (Part 3) K=8",
    ],
]
df_out.to_csv("dev.p3.8th.out", sep=" ", index=False, header=False)


# ## Part 4

# ### Version 1

# In[53]:


import numpy as np


def improved_viterbi_algorithm_1(
    sent, transition_probs, emission_probs, tags, all_states, suffix_probs
):
    """
    Use the Viterbi algorithm with suffix-based handling of unknown words.

    Parameters:
    - sent: list of tokens (words) in the sentence
    - transition_probs: dict of transition probabilities
    - emission_probs: dict of emission probabilities
    - tags: list of possible tags
    - all_states: list of all known tokens
    - suffix_probs: dict of probabilities based on word suffixes

    Returns:
    - best_path: list of predicted tags for the sentence
    - best_path_prob: probability of the best path
    """

    # Initialize the viterbi matrix with zeros. Dimensions: [number of tags, length of sentence]
    viterbi = np.zeros((len(tags), len(sent)))

    # Initialize the backpointer matrix with zeros. This matrix will help backtrack the best path later.
    backpointer = np.zeros((len(tags), len(sent)), "int")

    # Initialization step for the first word of the sentence
    for i, tag in enumerate(tags):
        # If the first word is unknown (not in all_states)...
        if sent[0] not in all_states:
            # Use suffix-based estimation for the tag
            estimated_tag = None
            for length in range(
                3, 0, -1
            ):  # Check 3-letter, 2-letter, then 1-letter suffixes
                if len(sent[0]) >= length and sent[0][-length:] in suffix_probs:
                    estimated_tag = suffix_probs[sent[0][-length:]]
                    break

            # Update the viterbi matrix based on whether a suffix match was found
            if estimated_tag:
                viterbi[i, 0] = (
                    transition_probs[("START", tag)] if tag == estimated_tag else 1e-10
                )
            else:
                # If no suffix match, assign a small probability to avoid zero probability
                viterbi[i, 0] = transition_probs[("START", tag)] * 1e-10
        else:
            # If the word is known, simply use the transition and emission probabilities
            viterbi[i, 0] = transition_probs[("START", tag)] * emission_probs.get(
                (tag, sent[0]), 1e-10
            )

    # Recursion step: fill in the rest of the viterbi matrix
    for t in range(1, len(sent)):
        for i, tag in enumerate(tags):
            max_prob, best_tag = max(
                (
                    viterbi[j, t - 1]
                    * transition_probs[(tags[j], tag)]
                    * emission_probs.get((tag, sent[t]), 1e-10),
                    j,
                )
                for j in range(len(tags))
            )
            viterbi[i, t] = max_prob
            backpointer[i, t] = best_tag

    # Termination step: determine the most probable last tag and its probability
    best_path_pointer = np.argmax(
        [viterbi[i, -1] * transition_probs[(tags[i], "STOP")] for i in range(len(tags))]
    )
    best_path_prob = (
        viterbi[best_path_pointer, -1]
        * transition_probs[(tags[best_path_pointer], "STOP")]
    )

    # Backtrack through the backpointer matrix to determine the best path (most probable tag sequence)
    best_path = []
    for t in range(len(sent) - 1, -1, -1):
        best_path.insert(0, tags[best_path_pointer])
        best_path_pointer = backpointer[best_path_pointer, t]

    return best_path, best_path_prob


# In[54]:


from collections import defaultdict


def dataframe_to_list(df):
    """
    Convert a DataFrame with columns 'Token' and 'Label' into a list of (Token, Label) tuples.

    Parameters:
    - df: DataFrame with columns 'Token' and 'Label'.

    Returns:
    - list of (Token, Label) tuples.
    """
    return [(token, label) for token, label in zip(df["Token"], df["Label"])]


def generate_suffix_probs(df, max_suffix_length=3):
    """
    Generate a dictionary of most common tags for each word suffix observed in the training data.

    Parameters:
    - df: DataFrame containing training data with columns 'Token' and 'Label'.
    - max_suffix_length: maximum length for word suffixes to consider (default is 3).

    Returns:
    - suffix_probs: a dictionary where keys are suffixes and values are their most common tags.
    """
    # Convert the dataframe to a list of (Token, Label) tuples for easier processing
    training_data = dataframe_to_list(df)
    # Use nested defaultdict to count occurrences of each tag for each suffix
    suffix_tag_counts = defaultdict(lambda: defaultdict(int))

    # Iterate through training data to populate the suffix_tag_counts dictionary
    for word, tag in training_data:
        for i in range(1, max_suffix_length + 1):
            suffix = word[-i:]
            suffix_tag_counts[suffix][tag] += 1

    # Dictionary to store the most common tag for each suffix
    suffix_probs = {}

    # For each suffix, determine the most common tag and store in suffix_probs
    for suffix, tag_counts in suffix_tag_counts.items():
        most_common_tag = max(tag_counts, key=tag_counts.get)
        suffix_probs[suffix] = most_common_tag

    return suffix_probs


# In[100]:


# Generate the suffix probabilities using the dataframe
suffix_prob = generate_suffix_probs(df.copy())
print(suffix_prob)


# In[56]:


# List to store the best paths for each sentence in 'sent'
all_best_path = []
# List to store the probabilities of each best path
all_best_path_prob = []

# Iterate over each sentence in 'sent'
for i in sent:
    # Use the improved Viterbi algorithm (version 1) to get the best path and its probability
    # for the current sentence 'i'
    best_path, best_path_prob = improved_viterbi_algorithm_1(
        i, trans_probs, emit_probs, tags, all_states, suffix_prob
    )
    # Append an empty string to 'best_path' to match the sentence length
    best_path += [""]
    # Add the best path for this sentence to the overall list of best paths
    all_best_path += best_path
    # Add the probability of this best path to the overall list of best path probabilities
    all_best_path_prob += [best_path_prob]

# Print the overall best paths and their probabilities for all sentences
print(all_best_path)
print(all_best_path_prob)


# In[57]:


# Add the predicted labels to the 'df_test' DataFrame
df_test["Predicted Label (Part 4) Ver. 1"] = all_best_path
# Display the updated 'df_test' DataFrame
display(df_test)


# In[58]:


df_out = df_test.loc[:, ["Token", "Predicted Label (Part 4) Ver. 1"]]
# df_out.to_csv('EvalScript/dev.prediction4_1', sep=' ', index=False, header=False)


# ### Version 2

# In[59]:


import numpy as np


def improved_viterbi_algorithm_2(
    sent, transition_probs, emission_probs, tags, all_states, suffix_probs, prefix_probs
):
    """
    Implements an improved version of the Viterbi algorithm that uses both suffixes and prefixes
    to handle unknown words.

    Parameters:
    - sent: A sequence of observations (words in a sentence).
    - transition_probs: A dictionary specifying the transition probabilities between states (tags).
    - emission_probs: A dictionary specifying the probabilities of observations given states.
    - tags: A set of possible hidden states (e.g., part-of-speech tags).
    - all_states: A set of all words that have been trained on.
    - suffix_probs: A dictionary mapping known suffixes to their most probable tag.
    - prefix_probs: A dictionary mapping known prefixes to their most probable tag.

    Returns:
    - best_path: The most probable sequence of tags.
    - best_path_prob: The probability of the most probable sequence of tags.
    """

    # Initialize matrices to store probabilities and backpointers.
    viterbi = np.zeros((len(tags), len(sent)))
    backpointer = np.zeros((len(tags), len(sent)), "int")

    # Initialization step: Compute probabilities for the first observation.
    for i, tag in enumerate(tags):
        # Handle unknown words using suffixes and prefixes.
        if sent[0] not in all_states:
            estimated_tag = None
            # Check suffixes and prefixes (from longer to shorter) to estimate the tag.
            for length in range(3, 0, -1):
                if len(sent[0]) >= length:
                    suffix = sent[0][-length:]
                    prefix = sent[0][:length]
                    if suffix in suffix_probs:
                        estimated_tag = suffix_probs[suffix]
                        break
                    elif prefix in prefix_probs:
                        estimated_tag = prefix_probs[prefix]
                        break

            # Assign probabilities based on the estimated tag.
            if estimated_tag:
                viterbi[i, 0] = (
                    transition_probs[("START", tag)] if tag == estimated_tag else 1e-10
                )
            else:
                viterbi[i, 0] = transition_probs[("START", tag)] * 1e-10
        else:  # For known words, compute the probability as usual.
            viterbi[i, 0] = transition_probs[("START", tag)] * emission_probs.get(
                (tag, sent[0]), 1e-10
            )

    # Recursion step: Compute the probabilities for subsequent observations.
    for t in range(1, len(sent)):
        for i, tag in enumerate(tags):
            max_prob, best_tag = max(
                (
                    viterbi[j, t - 1]
                    * transition_probs[(tags[j], tag)]
                    * emission_probs.get((tag, sent[t]), 1e-10),
                    j,
                )
                for j in range(len(tags))
            )
            viterbi[i, t] = max_prob
            backpointer[i, t] = best_tag

    # Termination step: Determine the most probable final tag.
    best_path_pointer = np.argmax(
        [viterbi[i, -1] * transition_probs[(tags[i], "STOP")] for i in range(len(tags))]
    )
    best_path_prob = (
        viterbi[best_path_pointer, -1]
        * transition_probs[(tags[best_path_pointer], "STOP")]
    )

    # Backtrack to construct the most probable sequence of tags.
    best_path = []
    for t in range(len(sent) - 1, -1, -1):
        best_path.insert(0, tags[best_path_pointer])
        best_path_pointer = backpointer[best_path_pointer, t]

    return best_path, best_path_prob


# In[60]:


from collections import defaultdict


def generate_prefix_probs(df, max_prefix_length=3):
    # Convert the DataFrame 'df' into a list of tuples (token, label)
    training_data = dataframe_to_list(df)
    # Define a nested defaultdict to store counts of each prefix-tag pair
    prefix_tag_counts = defaultdict(lambda: defaultdict(int))
    # Dictionary to store the most common tag associated with each prefix
    prefix_probs = {}
    # Count prefix-tag pairs
    for word, tag in training_data:
        # Iterate through each prefix length from 1 to 'max_prefix_length'
        for i in range(1, max_prefix_length + 1):
            # Extract prefix of length 'i' from the word
            prefix = word[:i]
            # Increment the count for the given prefix and tag combination
            prefix_tag_counts[prefix][tag] += 1
    # Determine most common tag for each prefix
    for prefix, tag_counts in prefix_tag_counts.items():
        # Identify the tag with the maximum count for the current prefix
        most_common_tag = max(tag_counts, key=tag_counts.get)
        # Store the most common tag for the prefix in the 'prefix_probs' dictionary
        prefix_probs[prefix] = most_common_tag
    # Return the 'prefix_probs' dictionary containing prefix and its most common associated tag
    return prefix_probs


# In[61]:


# Generate prefix probabilities for the given DataFrame 'df'
prefix_prob = generate_prefix_probs(df.copy())
# Print the generated prefix probabilities
print(prefix_prob)


# In[62]:


# Initialize lists to store the best path (sequence of tags) and its probability for each sentence in the test set
all_best_path = []
all_best_path_prob = []

# Iterate over each sentence in the test set
for i in sent:
    # Use the improved Viterbi algorithm version 2 to predict the best path (sequence of tags) for the sentence
    # This version of the algorithm utilizes both suffix and prefix probabilities for better prediction of unknown words
    best_path, best_path_prob = improved_viterbi_algorithm_2(
        i, trans_probs, emit_probs, tags, all_states, suffix_prob, prefix_prob
    )
    # Append a blank label to the end of the predicted sequence (for alignment with the original data format)
    best_path += [""]
    # Add the predicted sequence and its probability to the lists
    all_best_path += best_path
    all_best_path_prob += [best_path_prob]

# Display the predicted sequences and their probabilities
print(all_best_path)
print(all_best_path_prob)


# In[63]:


# Add the predicted labels to the test DataFrame as a new column "Predicted Label (Part 4) Ver. 2"
df_test["Predicted Label (Part 4) Ver. 2"] = all_best_path
# Display the updated test DataFrame with the new predictions
display(df_test)


# In[64]:


df_out = df_test.loc[:, ["Token", "Predicted Label (Part 4) Ver. 2"]]
# df_out.to_csv('EvalScript/dev.prediction4_2', sep=' ', index=False, header=False)
df_out.to_csv("dev.p4.out", sep=" ", index=False, header=False)


# In[107]:


# Open the file 'test_in.txt' in read mode
with open("test_in.txt", "r") as f:
    # Read all lines from the file and store them in a list
    lines = f.readlines()

# Remove any trailing and leading whitespaces (like newline characters) from each line
data = [line.strip() for line in lines]

# Convert the list of cleaned lines into a pandas DataFrame with a single column named 'Token'
df_test_test = pd.DataFrame(data, columns=["Token"])
# Create observed sequences based on the test data.
sent_test = create_observed_states(df_test_test)


# In[108]:


display(df_test_test)
print(sent_test)


# In[109]:


# Generate prefix probabilities for the given DataFrame 'df_test'
prefix_prob_test = generate_prefix_probs((df_test.loc[:, ["Token", "Label"]]).copy())
# Print the generated prefix probabilities
print(prefix_prob_test)

# Generate the suffix probabilities using the dataframe
suffix_prob_test = generate_suffix_probs((df_test.loc[:, ["Token", "Label"]]).copy())
# Print the generated suffix probabilities
print(suffix_prob_test)


# In[110]:


# Initialize lists to store the best path (sequence of tags) and its probability for each sentence in the test set
all_best_path_test = []
all_best_path_prob_test = []

# Iterate over each sentence in the test set
for i in sent_test:
    # Use the improved Viterbi algorithm version 2 to predict the best path (sequence of tags) for the sentence
    # This version of the algorithm utilizes both suffix and prefix probabilities for better prediction of unknown words
    best_path_test, best_path_prob_test = improved_viterbi_algorithm_2(
        i,
        trans_probs,
        emit_probs,
        tags,
        all_states,
        {**suffix_prob, **suffix_prob_test},
        {**prefix_prob, **prefix_prob_test},
    )
    # Append a blank label to the end of the predicted sequence (for alignment with the original data format)
    best_path_test += [""]
    # Add the predicted sequence and its probability to the lists
    all_best_path_test += best_path_test
    all_best_path_prob_test += [best_path_prob_test]

# Display the predicted sequences and their probabilities
print(all_best_path_test)
print(all_best_path_prob_test)


# In[111]:


# Add the computed best paths as a new column to the test DataFrame.
df_test_test["Predicted Label"] = all_best_path_test
display(df_test_test)


# In[112]:


df_out_test = df_test_test.loc[:, ["Token", "Predicted Label"]]
df_out_test.to_csv("test.p4.out", sep=" ", index=False, header=False)

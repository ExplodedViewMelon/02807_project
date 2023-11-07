# %%
import mmh3
import sys
import pickle
import pandas as pd
import numpy as np
from dataloader import CitationDataset
from typing import List
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=10)


# %%
# dataset = CitationDataset()
# df = dataset.load_dataframe(subset=True)
# df.to_pickle("df.pkl")

df = pd.read_pickle("df.pkl")



# %%
df["abstract"][df["abstract"] != ""].sort_values()

# %%
def shingle(text: str, shingle_size):
    text_list = text.split()
    return list(set(" ".join(text_list[i:i+shingle_size]) for i in range(len(text_list)-shingle_size+1)))

def minhash(text_list, seed) -> int:
    hash_list = [mmh3.hash(shingle, seed) for shingle in text_list] 
    return min(hash_list)

def clean_text(aString):
    output = aString.replace('\n','')
    output_list = output.split()
    output_list = [''.join(ch for ch in aWord if ch.isalnum()) for aWord in output_list]
    output_list = [s.lower() for s in output_list]
    output = ' '.join(output_list)
    return " ".join(output.split())

def get_signature(text: str, shingle_size = 3, sig_len = 5) -> List[int]:    
    shingle_list = shingle(text, shingle_size)
    if len(shingle_list) == 0:
        return pd.NA
    try:
        signature = [minhash(shingle_list, seed) for seed in range(sig_len)]
    except  Exception as e:
        print(text)
        print(shingle_list)
        sys.exit(e)
    return signature


test_string = "this is a test string to shingle and hash"

print(get_signature(test_string))
    

test2 = "theemergingtrendsinflossresearchanddevelopmentworkshopserieswillbebasedonthegrowinginterestofresearchersandpractitionersinfreelibreopensourcesoftwareflossthefirstworkshopwillbespecificallyfocusedondiscussingthephenomenonofglobalflossdevelopmentandhowtoimprovecllaborationandthecommunicationofresultsbetweenresearcherspractitionersandflosscommunitiesforthispurposetheoverarchingthemeofthisyearsworkshopisfeedingbackthecommunitiesitsgoalistobringtogetheracademicresearchersindustrymembersandflossdevelopersandtodiscusscrossfertilizationofresultsonflossresearchandpractice"
print(shingle(test2, 3))

# %%
df["abstract"] = df["abstract"].parallel_apply(clean_text)



# %%
df["signature"] = df["abstract"][df["abstract"] != ""].parallel_apply(get_signature)

# %%
df

# %%




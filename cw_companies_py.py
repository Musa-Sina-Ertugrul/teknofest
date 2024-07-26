# %%
import pandas as pd
import numpy as np

# %%
companies = pd.read_csv("/home/musasina/projects/teknofest/msnet/datasets/companies.csv").dropna()

# %%
companies = companies.convert_dtypes(convert_string=True)

# %%
companies_array = np.array(companies).flatten()

# %%
df_train = pd.read_csv("/home/musasina/projects/teknofest/msnet/datasets/train_turkish.csv")

# %%
def cut_text(text:str):
    splitted = str(text).split()
    splitted = splitted[:15]
    return " ".join(splitted)

# %%
df_train["text"] = df_train["text"].map(cut_text)

# %%
df_train_texts = df_train["text"].to_list()

# %%
labels = {"Notr":1,"Positive":2,"Negative":3}

# %%
sentiment = df_train["label"].map(str).to_numpy()

# %%
sentiment[0]

# %%
sentiment = np.apply_along_axis(lambda x : labels[x[0]],axis=1,arr=sentiment.reshape((-1,1)))

# %%
notr_bow = pd.read_csv("/home/musasina/projects/teknofest/msnet/bows/notr_bow.csv")
positive_bow = pd.read_csv("/home/musasina/projects/teknofest/msnet/bows/positive_bow.csv")
negative_bow = pd.read_csv("/home/musasina/projects/teknofest/msnet/bows/negative_bow.csv")

# %%
notr_bow_set = set(notr_bow["words"].map(str).to_list())
positive_bow_set = set(positive_bow["words"].map(str).to_list())
negative_bow_set = set(negative_bow["words"].map(str).to_list())

# %%
texts_with_companies = []

for text,label in zip(df_train_texts,sentiment):
    for _ in range(5):
        company = np.random.choice(companies_array)
        splitted_text = text.split()
        labels = [0]*15
        index = np.random.randint(0,len(splitted_text))
        for i,word in enumerate(splitted_text):
            if word in notr_bow_set:
                labels[i]=1
            elif word in positive_bow_set:
                labels[i]=2
            elif word in negative_bow_set:
                labels[i]=3
        labels.insert(index,label)
        splitted_text.insert(index,company)
        texts_with_companies.append({
            "text":" ".join(splitted_text),
            "labels":labels
        })

# %%
df_final = pd.DataFrame(texts_with_companies)
df_final.to_csv("/home/musasina/projects/teknofest/msnet/datasets/train_final.csv",index=False)



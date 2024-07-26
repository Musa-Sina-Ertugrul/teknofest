# %%
import pandas as pd
import sklearn
import sklearn.utils
from multiprocessing import Queue, Process
from tqdm.auto import tqdm


def func(queue: Queue, df: pd.DataFrame):
    def join_str(review):
        return " ".join(review)

    import logging

    from zemberek import (
        TurkishSentenceNormalizer,
        TurkishSentenceExtractor,
        TurkishMorphology,
        TurkishSpellChecker,
    )

    logger = logging.getLogger(__name__)

    morphology = TurkishMorphology.create_with_defaults()
    normalizer = TurkishSentenceNormalizer(morphology)
    extractor = TurkishSentenceExtractor()
    spell_checker = TurkishSpellChecker(morphology)

    # %%
    def correct_spells(review: str):

        words = review.split(" ")
        for i, word in enumerate(words):
            try:
                words[i] = spell_checker.suggest_for_word(word)[0]
            except BaseException:
                continue
        return " ".join(words)

    # %%
    df["review"] = df["review"].map(correct_spells)

    # %%
    def extract_sentence(review):
        return extractor.from_paragraph(review)

    # %%
    df["review"] = df["review"].map(extract_sentence)

    # %%
    df = df[df["review"] != ""]

    # %%
    def normalize_long_text(paragraph: str) -> str:
        result = []
        for sentence in paragraph:
            try:
                result.append(normalizer.normalize(sentence))
            except BaseException:
                continue
        return " ".join(result)

    # %%
    df["review"] = df["review"].map(normalize_long_text)

    # %% [markdown]
    # ### Stopwords

    # %%
    from nltk.corpus import stopwords
    import re
    import nltk

    nltk.download("stopwords")
    stops = set(stopwords.words("turkish"))
    print(stops)

    # %%
    def clear_stop_words(sentence):
        return [word for word in sentence.split(" ") if word not in stops]

    # %%
    df["review"] = df["review"].map(clear_stop_words)

    # %% [markdown]
    # ### Lemmatization

    # %%
    df["review"] = df["review"].map(join_str)

    # %%
    def remove_upper_punctuation(review):
        return review.replace('"', "").replace("’", "").replace("'", "").replace("”", "")

    # %%
    df["review"] = df["review"].map(remove_upper_punctuation)

    # %%
    def sep_text(review):
        return review.split(" ")

    # %%

    df["review"] = df["review"].map(sep_text)

    # %%
    import zeyrek

    analyzer = zeyrek.MorphAnalyzer()

    def lemmatize_sent(review):
        result = []
        for word in review:
            try:
                result.append(analyzer.lemmatize(word)[0][1][0])
            except BaseException:
                result.append(word)
        return " ".join(result)

    # %%
    df["review"] = df["review"].map(lemmatize_sent)

    # %%
    df["review"] = df["review"].map(lambda x: x.casefold())
    """
    import unidecode

    turkish_chars = "ÇçĞğıİÖöŞşÜü"
    normal_chars = unidecode.unidecode(turkish_chars)

    def change_turkish_chars(review:str):
        for char,turkish_char in zip(normal_chars,turkish_chars):
            review = review.replace(turkish_char,char)
        return review
    
    df["review"] = df["review"].map(change_turkish_chars)
    """
    queue.put(df)
    print("done")


if __name__ == "__main__":
    # %%
    # %%
    df_train = pd.read_csv(
        "/home/musasina/projects/teknofest/msnet/datasets/train_final.csv"
    )
    df_train.head()

    # %%
    df_train = df_train.convert_dtypes(convert_string=True)

    # %%
    df_train = df_train[df_train["review"] != ""]
    """
    # %%
    print(len(df_train[df_train["label"] == "Notr"]))
    print(len(df_train[df_train["label"] == "Positive"]))
    print(len(df_train[df_train["label"] == "Negative"]))

    # %%
    df_train_positive = df_train[df_train["label"] == "Positive"].sample(
        50905, random_state=42
    )
    df_train_negative = df_train[df_train["label"] == "Negative"].sample(
        50905, random_state=42
    )
    df_train_notr = df_train[df_train["label"] == "Notr"].sample(50905, random_state=42)
    df_train: pd.DataFrame = sklearn.utils.shuffle(
        pd.concat(
            (df_train_negative, df_train_positive, df_train_notr), ignore_index=True
        ),
        random_state=42,
    )
    """
    # %%
    #df_test.head()

    # %%
    #df_test = df_test.convert_dtypes(convert_string=True)

    # %% [markdown]
    # # review Preprocessing

    # %%
    import re

    # %% [markdown]
    # ### Convert to lower case

    # %%
    df_train["review"] = [token.casefold() for token in df_train["review"]]
    df_train.head(5)

    # %% [markdown]
    # ### Remove @ mentions and hyperlinks

    # %%
    found = df_train[df_train["review"].str.contains("@")]
    found.count()

    # %%
    df_train.info()

    # %%
    df_train["review"] = (
        df_train["review"]
        .replace("@[A-Za-z0-9]+", "", regex=True)
        .replace("@[A-Za-z0-9]+", "", regex=True)
    )
    found = df_train[df_train["review"].str.contains("@")]
    found.count()

    # %%
    found = df_train[df_train["review"].str.contains("http")]
    found.count()

    # %%
    df_train["review"] = (
        df_train["review"]
        .replace(r"http\S+", "", regex=True)
        .replace(r"www\S+", "", regex=True)
    )
    found = df_train[df_train["review"].str.contains("http")]
    found.count()

    # %% [markdown]
    # ### Remove Punctations & Emojies & Numbers

    # %%
    sentences = df_train["review"].copy()
    new_sent = []
    i = 0
    for sentence in sentences:
        new_sentence = re.sub("[0-9]+", "", sentence)
        new_sent.append(new_sentence)
        i += 1

    df_train["review"] = new_sent
    df_train["review"].head(5)

    # %%


    # %%
    df_train["review"] = new_sent
    df_train["review"].head(5)

    # %%
    def join_str(review):
        return " ".join(review)

    # %%
    df_train["review"] = df_train["review"].map(join_str)

    # %%
    df_train = df_train.convert_dtypes(convert_string=True)

    # %% [markdown]
    # # Zemberek-NLP

    # %% [markdown]
    # ## Tokenization

    # %% [markdown]
    # ### Sentence Normalization

    # %%
    # %%
    df_train_1 = df_train.iloc[:10000, :].copy()
    df_train_2 = df_train.iloc[10000:20000, :].copy()
    df_train_3 = df_train.iloc[20000:30000, :].copy()
    df_train_4 = df_train.iloc[30000:40000, :].copy()
    df_train_5 = df_train.iloc[40000:50000, :].copy()
    df_train_6 = df_train.iloc[50000:60000, :].copy()
    df_train_7 = df_train.iloc[60000:70000, :].copy()
    df_train_8 = df_train.iloc[70000:80000, :].copy()
    df_train_9 = df_train.iloc[80000:90000, :].copy()
    df_train_10 = df_train.iloc[90000:100000, :].copy()
    df_train_11 = df_train.iloc[100000:110000, :].copy()
    df_train_12 = df_train.iloc[110000:120000, :].copy()
    df_train_13 = df_train.iloc[120000:130000, :].copy()
    df_train_14 = df_train.iloc[130000:140000, :].copy()
    df_train_15 = df_train.iloc[150000:, :].copy()
    dfs = [
        df_train_1,
        df_train_2,
        df_train_3,
        df_train_4,
        df_train_5,
        df_train_6,
        df_train_7,
        df_train_8,
        df_train_9,
        df_train_10,
        df_train_11,
        df_train_12,
        df_train_13,
        df_train_14,
        df_train_15
    ]

    q = Queue()
    processes = []
    for i in range(15):
        p = Process(target=func, args=(q, dfs[i]))
        processes.append(p)
        p.start()
    dfs = []
    for _ in range(15):
        dfs.append(q.get())
        print(f"recieved{_}")
    for i in range(15):
        processes[i].join()
        # %% [markdown]
        # ### Remove Rare Words
        # %%

    df_train = pd.concat(dfs, ignore_index=True)

    freq = pd.Series((' '.join(df_train['review'])).split()).value_counts()
    less_freq = set(freq[freq == 1])
    df_train["review"] = df_train["review"].map(lambda x: " ".join(x for x in x.split() if x not in less_freq))

    # %%
    df_train.to_csv(
        "/home/musasina/projects/teknofest/msnet/datasets/train_final_turkish.csv", index=False
    )
    # df_test.to_csv("/home/musasina/projects/teknofest/gpt2/datasets/test.csv",index=False)

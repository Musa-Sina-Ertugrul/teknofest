# %%
import pandas as pd
import sklearn
import sklearn.utils
from multiprocessing import Queue, Process
from tqdm.auto import tqdm


def func(queue: Queue, df: pd.DataFrame):
    def join_str(text):
        return " ".join(text)

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
    def correct_spells(text: str):

        words = text.split(" ")
        for i, word in enumerate(words):
            try:
                words[i] = spell_checker.suggest_for_word(word)[0]
            except BaseException:
                continue
        return " ".join(words)

    # %%
    df["text"] = df["text"].map(correct_spells)

    # %%
    def extract_sentence(text):
        return extractor.from_paragraph(text)

    # %%
    df["text"] = df["text"].map(extract_sentence)

    # %%
    df = df[df["text"] != ""]

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
    df["text"] = df["text"].map(normalize_long_text)

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
    df["text"] = df["text"].map(clear_stop_words)

    # %% [markdown]
    # ### Lemmatization

    # %%
    df["text"] = df["text"].map(join_str)

    # %%
    def remove_upper_punctuation(text):
        return text.replace('"', "").replace("’", "").replace("'", "").replace("”", "")

    # %%
    df["text"] = df["text"].map(remove_upper_punctuation)

    # %%
    def sep_text(text):
        return text.split(" ")

    # %%

    df["text"] = df["text"].map(sep_text)

    # %%
    import zeyrek

    analyzer = zeyrek.MorphAnalyzer()

    def lemmatize_sent(text):
        result = []
        for word in text:
            try:
                result.append(analyzer.lemmatize(word)[0][1][0])
            except BaseException:
                result.append(word)
        return " ".join(result)

    # %%
    df["text"] = df["text"].map(lemmatize_sent)

    # %%
    df["text"] = df["text"].map(lambda x: x.casefold())
    """
    import unidecode

    turkish_chars = "ÇçĞğıİÖöŞşÜü"
    normal_chars = unidecode.unidecode(turkish_chars)

    def change_turkish_chars(text:str):
        for char,turkish_char in zip(normal_chars,turkish_chars):
            text = text.replace(turkish_char,char)
        return text
    
    df["text"] = df["text"].map(change_turkish_chars)
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
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df_train = pd.read_csv("hf://datasets/winvoker/turkish-sentiment-analysis-dataset/" + splits["train"])

    # %%
    df_train = df_train.convert_dtypes(convert_string=True)

    # %%
    df_train = df_train[df_train["text"] != ""]
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
    # # text Preprocessing

    # %%
    import re

    # %% [markdown]
    # ### Convert to lower case

    # %%
    df_train["text"] = [token.casefold() for token in df_train["text"]]
    df_train.head(5)

    # %% [markdown]
    # ### Remove @ mentions and hyperlinks

    # %%
    found = df_train[df_train["text"].str.contains("@")]
    found.count()

    # %%
    df_train.info()

    # %%
    df_train["text"] = (
        df_train["text"]
        .replace("@[A-Za-z0-9]+", "", regex=True)
        .replace("@[A-Za-z0-9]+", "", regex=True)
    )
    found = df_train[df_train["text"].str.contains("@")]
    found.count()

    # %%
    found = df_train[df_train["text"].str.contains("http")]
    found.count()

    # %%
    df_train["text"] = (
        df_train["text"]
        .replace(r"http\S+", "", regex=True)
        .replace(r"www\S+", "", regex=True)
    )
    found = df_train[df_train["text"].str.contains("http")]
    found.count()

    # %% [markdown]
    # ### Remove Punctations & Emojies & Numbers

    # %%
    sentences = df_train["text"].copy()
    new_sent = []
    i = 0
    for sentence in sentences:
        new_sentence = re.sub("[0-9]+", "", sentence)
        new_sent.append(new_sentence)
        i += 1

    df_train["text"] = new_sent
    df_train["text"].head(5)

    # %%


    # %%
    df_train["text"] = new_sent
    df_train["text"].head(5)

    # %%
    def join_str(text):
        return " ".join(text)

    # %%
    df_train["text"] = df_train["text"].map(join_str)

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
    df_train_1 = df_train.iloc[:30_000, :].copy()
    df_train_2 = df_train.iloc[30_000:60_000, :].copy()
    df_train_3 = df_train.iloc[60_000:90_000, :].copy()
    df_train_4 = df_train.iloc[90_000:120_000, :].copy()
    df_train_5 = df_train.iloc[120_000:150_000, :].copy()
    df_train_6 = df_train.iloc[150_000:180_000, :].copy()
    df_train_7 = df_train.iloc[180_000:210_000, :].copy()
    df_train_8 = df_train.iloc[210_000:240_000, :].copy()
    df_train_9 = df_train.iloc[240_000:270_000, :].copy()
    df_train_10 = df_train.iloc[270_000:300_000, :].copy()
    df_train_11 = df_train.iloc[300_000:330_000, :].copy()
    df_train_12 = df_train.iloc[330_000:360_000, :].copy()
    df_train_13 = df_train.iloc[360_000:390_000, :].copy()
    df_train_14 = df_train.iloc[390_000:420_000, :].copy()
    df_train_15 = df_train.iloc[420_000:, :].copy()
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

    freq = pd.Series((' '.join(df_train['text'])).split()).value_counts()
    less_freq = set(freq[freq == 1])
    df_train["text"] = df_train["text"].map(lambda x: " ".join(x for x in x.split() if x not in less_freq))

    # %%
    df_train.to_csv(
        "/home/musasina/projects/teknofest/msnet/datasets/train_turkish_all.csv", index=False
    )
    # df_test.to_csv("/home/musasina/projects/teknofest/gpt2/datasets/test.csv",index=False)

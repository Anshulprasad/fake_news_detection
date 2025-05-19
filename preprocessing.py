import re, string, joblib, pandas as pd

from scipy.sparse import csr_matrix
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

from config import VECTORIZER_PATH


def wordopt(text):
    """
        Cleans the input text by performing the following operations:
    - Converts text to lowercase.
    - Removes text inside square brackets.
    - Removes URLs.
    - Removes HTML tags.
    - Removes punctuation.
    - Removes newline characters.
    - Removes words containing numbers.
    - Strips extra spaces.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\[.*?\]", "", text)  # Remove text inside square brackets
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>+", "", text)  # Remove HTML tags
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation), "", text
    )  # Remove punctuation
    text = re.sub(r"\n", "", text)  # Remove newline characters
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words containing numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def stem(df):
    """
    Applies stemming to the 'text' column of the input DataFrame using the Snowball Stemmer.
    Stemming reduces words to their root form.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'text' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'text' column stemmed.
    """
    stemmer = SnowballStemmer("english")
    df["text"] = df["text"].apply(
        lambda x: " ".join(stemmer.stem(word) for word in x.split())
    )

    logging.info("stemming completed!\n")
    return df


def shufling(df):
    """
    Shuffles the rows of the input DataFrame randomly and resets the index.

    Args:
        df (pd.DataFrame): The input DataFrame to be shuffled.

    Returns:
        pd.DataFrame: The shuffled DataFrame with reset index.
    """
    df = df.sample(frac=1, random_state=42)
    df.reset_index(drop=True, inplace=True)

    logging.info("shufling completed!\n")
    return df


def data_scaling(xv_train, xv_test):
    """
    Scales the input training and testing feature matrices to a range of [0, 1] using MinMaxScaler.
    Converts the scaled matrices back to sparse format.

    Args:
        xv_train (csr_matrix): The training feature matrix in sparse format.
        xv_test (csr_matrix): The testing feature matrix in sparse format.

    Returns:
        tuple: A tuple containing the scaled training and testing feature matrices in sparse format.
    """
    scaler = MinMaxScaler()
    xv_train = scaler.fit_transform(xv_train.toarray())
    xv_test = scaler.transform(xv_test.toarray())

    xv_train = csr_matrix(xv_train)
    xv_test = csr_matrix(xv_test)

    logging.info("scaling completed!\n")
    return xv_train, xv_test


def add_label(df_real, df_fake):
    """
    Adds a 'target' column to the input DataFrames to label real and fake news.
    - Real news is labeled as 1.
    - Fake news is labeled as 0.
    Concatenates the two DataFrames into a single DataFrame.

    Args:
        df_real (pd.DataFrame): The DataFrame containing real news.
        df_fake (pd.DataFrame): The DataFrame containing fake news.

    Returns:
        pd.DataFrame: The concatenated DataFrame with labeled data.
    """

    df_real["target"] = 1
    df_fake["target"] = 0

    df = pd.concat([df_real, df_fake], axis=0)

    logging.info("labeling and concatination completed!\n")
    return df


def text_to_vector(x_train, x_test):
    """
    Converts the 'text' column of the input training and testing DataFrames into TF-IDF vectors.
    Saves the fitted TfidfVectorizer to a file for later use.

    Args:
    x_train (pd.DataFrame): The training DataFrame containing a 'text' column.
    x_test (pd.DataFrame): The testing DataFrame containing a 'text' column.

    Returns:
    tuple: A tuple containing the TF-IDF vectorized training and testing feature matrices.
    """

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

    xv_train = vectorizer.fit_transform(x_train["text"])
    xv_test = vectorizer.transform(x_test["text"])

    joblib.dump(vectorizer, VECTORIZER_PATH)

    xv_train = xv_train.astype("float32")
    xv_test = xv_test.astype("float32")

    logging.info("text to vector conversion completed!\n")
    return xv_train, xv_test


def train_test_spliting(df):
    """
    Splits the input DataFrame into training and testing sets.
    Separates the 'target' column as the label and the remaining columns as features.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and the 'target' column.

    Returns:
        tuple: A tuple containing the training and testing feature DataFrames and their corresponding labels.
    """
    y = df["target"]
    x = df.drop(columns=["target"])

    logging.info("split completed!\n")
    return train_test_split(x, y, test_size=0.33, random_state=42)

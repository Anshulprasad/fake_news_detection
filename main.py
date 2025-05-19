from config import (
    TRUE_CSV_PATH,
    FAKE_CSV_PATH,
    SOURCE_PATH,
    TARGET_PATH,
    MODEL_MNB_PATH,
    MODEL_RFC_PATH,
    MODEL_LR_PATH,
    MODEL_NN_PATH,
    MODEL_ABC_PATH,
    MODEL_L_PATH,
    NEWS_ARTICLES_PATH,
    TEST_SET_MANUAL,
    SCRAPED_ARTICLES
)
from data_collection import install_data, news_api, web_scraping_news
from preprocessing import (
    add_label,
    wordopt,
    stem,
    shufling,
    train_test_spliting,
    text_to_vector,
    data_scaling,
)
from train_classical import train, neural_network, voting_classifier
# from train_bert import bert

import pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

logging.info("\n")


def main():

    # download datasets from kaggle though kagglehub
    install_data(SOURCE_PATH, TARGET_PATH)

    try:
        # import data
        df_real = pd.read_csv(TRUE_CSV_PATH)
        df_fake = pd.read_csv(FAKE_CSV_PATH)

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # get labeled and combined dataset
    df = add_label(df_real, df_fake)

    # remove irrelevant features
    df.drop(columns=["title", "subject", "date"], inplace=True)
    logging.info("remove features completed!\n")

    # clean data
    df["text"] = df["text"].apply(wordopt)
    logging.info("cleaning completed!\n")

    # stemm using SnowballStemmer
    df = stem(df)

    # data shufling
    df = shufling(df)

    # split dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_spliting(df)

    # Save test set for manual testing
    test_set = x_test.copy()
    test_set['target'] = y_test
    test_set.to_csv(TEST_SET_MANUAL, index=False)
    logging.info("Test set saved to test_set_manual.csv for manual testing.\n")

    # text to vector conversion
    xv_train, xv_test = text_to_vector(x_train, x_test)

    # scaling
    xv_train, xv_test = data_scaling(xv_train, xv_test)

    # training models
    mnb = train(MultinomialNB, xv_train, xv_test, y_train, y_test)
    rfc = train(RandomForestClassifier, xv_train, xv_test, y_train, y_test)
    abc = train(AdaBoostClassifier, xv_train, xv_test, y_train, y_test)
    lr = train(LogisticRegression, xv_train, xv_test, y_train, y_test)
    nn = neural_network(xv_train, xv_test, y_train, y_test)
    l = voting_classifier(xv_train, xv_test, y_train, y_test)

    # save models
    joblib.dump(mnb, MODEL_MNB_PATH)
    joblib.dump(rfc, MODEL_RFC_PATH)
    joblib.dump(abc, MODEL_ABC_PATH)
    joblib.dump(lr, MODEL_LR_PATH)
    joblib.dump(nn, MODEL_NN_PATH)
    joblib.dump(l, MODEL_L_PATH)

    plt.show()


if __name__ == "__main__":
    main()

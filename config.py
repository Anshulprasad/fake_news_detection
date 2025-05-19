import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved")
os.makedirs(SAVED_DIR, exist_ok=True)

# Dataset paths
TRUE_CSV_PATH = os.path.join(BASE_DIR, 'fake-and-real-news-dataset', 'True.csv')
FAKE_CSV_PATH = os.path.join(BASE_DIR, 'fake-and-real-news-dataset', 'Fake.csv')
SOURCE_PATH = 'clmentbisaillon/fake-and-real-news-dataset'
TARGET_PATH = BASE_DIR

# Model save paths
MODEL_MNB_PATH = os.path.join(SAVED_DIR, 'model_mnb.pkl')
MODEL_RFC_PATH = os.path.join(SAVED_DIR, 'model_rfc.pkl')
MODEL_LR_PATH = os.path.join(SAVED_DIR, 'model_lr.pkl')
MODEL_NN_PATH = os.path.join(SAVED_DIR, 'model_nn.pkl')
MODEL_ABC_PATH = os.path.join(SAVED_DIR, 'model_abc.pkl')
MODEL_L_PATH = os.path.join(SAVED_DIR, 'model_l.pkl')

VECTORIZER_PATH = os.path.join(SAVED_DIR, "vectorizer.pkl")

MODEL_BERT_PATH = os.path.join(SAVED_DIR, 'model_bert.pkl')
TOKENIZER_BERT_PATH = os.path.join(SAVED_DIR, 'tokenizer_bert.pkl')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

NEWS_ARTICLES_PATH = os.path.join(SAVED_DIR, 'news_articles.csv')
TEST_SET_MANUAL = os.path.join(SAVED_DIR, 'test_set_manual.csv')
SCRAPED_ARTICLES = os.path.join(SAVED_DIR, 'bbc_scraped_articles.csv')

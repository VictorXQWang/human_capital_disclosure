import os
from multiprocessing import cpu_count

# Full path of the csv file containing the HC keyword list
HC_WL_CSV = r"D:\JIS_HC_Data_and_Code\Data\final_HC_wordlist.csv"

# Number of workers for parallel pre-processing; defaulted to the number of logical cores - 2
# i.e., 10 workers for a CPU with 6 physical cores supporting multiple threading
WORKERS: int = max(1, cpu_count()-2)

#Lemmatizer: "NLTK" or "SPACY" for using the NLTK or Spacy lemmatizer
LEMM: str = "NLTK"

# Chunk size of processed sentences to be saved to a disk file: 10000 for 10000 sentences
NUM_SENTS: int = 10000

# Folder for storing preprocessed inputs and phrase model
PHRASE_FOLDER = r"D:\temp\phrase"

# Folder for storing the w2v model
W2V_FOLDER = r"D:\temp\word2vec"

#Phraser parameters:
MIN_COUNT: int =10
THRESHOLD: int = 10
PROGRESS_PER: int = 1000000

# Full path of the EXCEL file containing seedwords
# Each column contains the seed words for one category
# The column header is the category name
SEEDWORD_CSV = r"D:\JIS_HC_Data_and_Code\Data\HC_seedwords.csv"

# Word2vec model parameters:
w2v_VECTOR_SIZE: int = 100
w2v_WINDOW: int = 5
w2v_MIN_COUNT: int = 5
w2v_WORKERS: int = 10
w2v_EPOCHS: int = 5

# Similar words parameters
TOPN: int = None   # None or an integer, e.g., 300
MIN_SCORE: int = 0  # between -1 and 1
ABSOLUTE: bool = True # True or False; if True, return all those whose absolute(score)>=MIN_SCORE

# Create three subfolders named "unigram", "bigram", "trigram" in the "Phrase Folder" for
# storing processed sentences consisting of unigrams, bigrams, and trigrams
# Create a "model" subfolder for storing phrase model data
# Create the "word2vec" folder for storing the w2v model

UNIGRAM_DIR = os.path.join(PHRASE_FOLDER, "unigram")
BIGRAM_DIR = os.path.join(PHRASE_FOLDER, "bigram")
TRIGRAM_DIR = os.path.join(PHRASE_FOLDER, "trigram")
MODEL_DIR = os.path.join(PHRASE_FOLDER, "model")

for dir in [UNIGRAM_DIR, BIGRAM_DIR, TRIGRAM_DIR, MODEL_DIR, W2V_FOLDER]:
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)





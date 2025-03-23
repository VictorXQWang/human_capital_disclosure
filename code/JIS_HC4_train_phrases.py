import os
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
from multiprocessing import Pool
import tqdm
import logging
import JIS_HC0_config as CONFIG
import JIS_HC1_util as UT
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# Generate preprocessed sentences from a single txtfile
def get_normalized_tokenized_sents_from_txtfile(txtfile):
    lemm_sents = UT.get_tokenized_lemmatized_sents_from_txtfile(txtfile)

    # Convert tokens containing numbers or punctuations into "1"/"," to reduce the vocabulary size
    # Those on the whitelist are bypassed
    normalized_sents = [[UT.normalize_token(t) for t in s] for s in lemm_sents]
    return normalized_sents

# Generate preprocessed sentences from txtfiles using multiple workers
def get_normalized_tokenized_sents_from_txtfiles(txtfiles, workers=CONFIG.WORKERS, chunksize=10):

    pool = Pool(processes=workers)
    # Quicker by ignoring sequence
    results = tqdm.tqdm(pool.imap_unordered(
        get_normalized_tokenized_sents_from_txtfile, txtfiles, chunksize=chunksize), total=len(txtfiles)
    )
    for sents in results:
        for sent in sents:
            yield sent

# Get phrases, whose components are all valid words
def get_valid_phrases(bigram_model):
    bigrams = list(bigram_model.export_phrases().keys())
    bigrams = [t for t in bigrams if UT.is_valid_ngram(t)]
    return bigrams

# Step 1: Generate unigram sentences
def get_unigram_sents(txtfiles, target_dir=CONFIG.UNIGRAM_DIR):
    pkl_files = UT.get_files_from_dir(target_dir, file_ext=".pkl")
    if len(pkl_files) > 0:
        print("Preprocessedd unigram sentences already exist.")
        print("To re-generate preprocessed sentences, delete all files in the 'unigram' folder and rerun the code.")
    else:
        print("Generating tokenized_sents_unigram", UT.current_timestamp())
        # This is a generator/Iterator and is powered by multiple-processing
        tokenized_sents = UT.Generator2Iterator(get_normalized_tokenized_sents_from_txtfiles, txtfiles)
        UT.save_sents_to_pkl_file(tokenized_sents, target_dir)
    print("Loading tokenized_sents_unigram", UT.current_timestamp())
    sentences = UT.Generator2Iterator(UT.load_sents_from_pkl_files, paras=target_dir)
    return sentences  # This a generator/iterator

# Step 2: Generate bigram sentences
def get_bigram_sents(txtfiles, bigram_model, unigram_dir=CONFIG.UNIGRAM_DIR, bigram_dir=CONFIG.BIGRAM_DIR):
    pkl_files = UT.get_files_from_dir(bigram_dir, file_ext=".pkl")
    if len(pkl_files) > 0:
        print("Preprocessed bigram sentences already exist.")
        print("To re-generate preprocessed sentences, delete everything in the 'bigram' folder and rerun the code.")
    else:
        print("Generating tokenized_sents_bigram", UT.current_timestamp())
        tokenized_sents_unigram = get_unigram_sents(txtfiles, unigram_dir)
        tokenized_sents_bigram = bigram_model[tokenized_sents_unigram]
        UT.save_sents_to_pkl_file(tokenized_sents_bigram, bigram_dir) # can use mp here
    print("Loading tokenized_sents_bigram", UT.current_timestamp())
    tokenized_sents_bigram = UT.Generator2Iterator(UT.load_sents_from_pkl_files, paras=bigram_dir)
    return tokenized_sents_bigram # This a generator/iterator

# Train the phrase model
def train_phrase_model(
        txtfiles, min_count=CONFIG.MIN_COUNT, threshold=CONFIG.THRESHOLD, progress_per=CONFIG.PROGRESS_PER
):

    print("Number of files to process: ", len(txtfiles), UT.current_timestamp())
    unigram_tokenized_sents = get_unigram_sents(txtfiles)

    cwd = os.getcwd()
    os.chdir(CONFIG.MODEL_DIR)

    print("Train the bigram model...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    bigram = Phrases(unigram_tokenized_sents, min_count=min_count, threshold=threshold, progress_per=progress_per)
    bigram_mod = Phraser(bigram)
    bigram_mod.save("bigram.model")

    # Generate tokenized sentences with bigrams
    bigram_tokenized_sents = get_bigram_sents(txtfiles, bigram_mod)

    # Feed tokenized sentences with bigrams to the phraser model to identify trigrams
    print("Train the trigram model...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    trigram = Phrases(bigram_tokenized_sents, min_count=min_count, threshold=threshold, progress_per=progress_per)
    trigram_mod = Phraser(trigram)
    trigram_mod.save("trigram.model")

    # Get bigrams from the entire corpus:
    print("Get valid bigrams")
    valid_bigrams = get_valid_phrases(bigram)

    print("Get valid trigrams")
    # This includes both trigrams and bigrams:
    valid_ngrams = get_valid_phrases(trigram)

    # Merge the two and drop duplicates
    valid_ngrams = set(valid_bigrams + valid_ngrams)
    valid_ngrams = list(valid_ngrams)
    UT.save_pickle(valid_ngrams, "valid_ngrams.pkl")

    # Output to the bigrams and trigrams into a CSV file for manual inspection if necessary
    df = pd.DataFrame({"Ngrams" : sorted(valid_ngrams)})
    df["Trigram"] = df.Ngrams.apply(lambda x: 1 if x.count("_") >= 2 else 0)
    print("Number of valid bigrams:", len(df[df.Trigram==0]))
    print("Number of valid trigrams:", len(df[df.Trigram==1]))
    # You can open this CSV file and inspect the valid phrases
    df.to_csv(r"valid_ngrams.csv", index=False)
    os.chdir(cwd)
    return trigram_mod, bigram_mod, valid_ngrams

if __name__ == "__main__":
    source_dir = r"D:\sample_proxy_statements"
    txtfiles = UT.get_files_from_dir(source_dir)
    m = train_phrase_model(txtfiles)

























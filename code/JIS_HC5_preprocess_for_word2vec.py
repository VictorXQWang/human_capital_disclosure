import os
import gensim
from multiprocessing import Pool
import tqdm
import logging
from functools import partial

import JIS_HC0_config as CONFIG
import JIS_HC1_util as UT
from JIS_HC4_train_phrases import train_phrase_model


# Load the phrase model and phrases if they already exist
def load_phrase_model(model_dir=CONFIG.MODEL_DIR):
    trigram_mod = bigram_mod = valid_ngrams = None
    cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        bigram_mod = gensim.models.phrases.Phraser.load("bigram.model")
        trigram_mod = gensim.models.phrases.Phraser.load("trigram.model")
        valid_ngrams = UT.load_pickle("valid_ngrams.pkl") # This is a list
    except:
        print("Phrase model does not exist.")
    os.chdir(cwd)
    return trigram_mod, bigram_mod, valid_ngrams

# Generate the phrase model (load the existing one if found)
def get_phrase_model(txtfiles, model_dir = CONFIG.MODEL_DIR):
     # First check if models already exist:
    trigram_mod, bigram_mod, valid_ngrams = load_phrase_model(model_dir=model_dir)

    # Check if all three objects are found
    if all([trigram_mod, bigram_mod, valid_ngrams]):
        print("Phrase model already exists.")
        print("To train a new phrase model, remove all files in the 'phrase' folder and rerun the code.")
    else:
        # Train a new model if not all three objects are found
        print("Train a new phrase model...")

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        trigram_mod, bigram_mod, valid_ngrams = train_phrase_model(txtfiles)
    return trigram_mod, bigram_mod, valid_ngrams

# Break those phrases not on the valid ngram list
def break_invalid_phrases(tokens_with_phrases, valid_ngrams):
    final_tokens=[]
    valid_ngrams = set(valid_ngrams)
    for t in tokens_with_phrases:
        # First bypass those on the whitelist
        if t in UT.white_list_ngram:
            final_tokens.append(t)
        # tagged as a phrase but does not meet the screening requirements
        if ("_" in t) and (t not in valid_ngrams):
            toks = t.split("_")
            final_tokens += toks
        else:
            final_tokens.append(t)
    return final_tokens

# Function for processing a single file from scratch
# Not used in this module, but may be useful if you want to adapt the code
def preprocess_sents_from_txtfile(txtfile, bigram_mod, trigram_mod, valid_ngrams):
    lemm_sents = UT.get_tokenized_lemmatized_sents_from_txtfile(txtfile)
    sents_with_bigrams = bigram_mod[lemm_sents]
    sents_with_trigrams = trigram_mod[sents_with_bigrams]
    sents_with_phrases = [break_invalid_phrases(s, valid_ngrams) for s in sents_with_trigrams]
    return sents_with_phrases

# Generate final phrased sentences by applying the trigram model to sentences with bigrams
def phrase_tokenized_bigram_sents(tokenized_bigram_sents, trigram_mod, valid_ngrams):
    # This includes numbers and punctuation
    # sent_tokens is a list of tokens, already tokenized
    tokenized_sents_with_trigrams = trigram_mod[tokenized_bigram_sents]
    phrased_tokenized_sents = [break_invalid_phrases(s, valid_ngrams) for s in tokenized_sents_with_trigrams]
    return phrased_tokenized_sents

# Generate cleaned sents with bigrams and trigrams (with numbers, puncs, and one-letter word removed)
# by using the preprocessed sents with bigrams from the Phrase training
def gen_phrased_sents_from_bigram_pkl_file(pkl_file, trigram_mod, valid_ngrams):
    bigram_sents = UT.load_pickle(pkl_file) # This is a list of tokenized sents
    trigram_sents = phrase_tokenized_bigram_sents(bigram_sents, trigram_mod, valid_ngrams)
    # Remove stopwords, numbers, etc., for faster training
    phrased_sents = [[t for t in s if UT.is_valid_word(t)] for s in trigram_sents]
    return phrased_sents, pkl_file  # This is a list of sents

# Use multiple workers to generate cleaned sents
def gen_phrased_sents_from_bigram_pkl_files_mp(pickle_files, trigram_mod, valid_ngrams, workers=CONFIG.w2v_WORKERS):
    pool = Pool(processes=workers)
    partial_func = partial(gen_phrased_sents_from_bigram_pkl_file, trigram_mod=trigram_mod, valid_ngrams=valid_ngrams)
    results = tqdm.tqdm(pool.imap_unordered(partial_func, pickle_files), total=len(pickle_files))
    for sents, basic_pkl_file in results:
        yield sents, os.path.basename(basic_pkl_file)

# Generate and save cleaned sents to the "trigram" folder
def gen_phrased_sents_and_save_to_disk(txtfiles, workers=CONFIG.WORKERS):

    # First check if trigram files and bigram files already exist and get the number of files
    num_trigram_files = num_bigram_files = 0
    if os.path.exists(CONFIG.TRIGRAM_DIR):
        trigram_files = UT.get_files_from_dir(CONFIG.TRIGRAM_DIR, file_ext=".pkl")
        num_trigram_files = len(trigram_files)

    if os.path.exists(CONFIG.BIGRAM_DIR):
        bigram_files = UT.get_files_from_dir(CONFIG.BIGRAM_DIR, file_ext=".pkl")
        num_bigram_files = len(bigram_files)

    # If trigram_files exist and their numbers are equal to those of bigram_files, then the data are complete:
    if (num_trigram_files == num_bigram_files) and num_trigram_files > 0:
        print("Preprocessed trigram files already exist.")
        print("To use a new phrase model, please remove all files in the 'phrase' folder and rerun the code")
        print("Final phrased sents for w2v training already generated.")
        print("Read sents from the 'trigram' folder...")
        return

    if num_bigram_files == 0:
        # If no bigram files for generating trigram files, then train a new model for generating the bigram files.
        trigram_mod, bigram_mod, valid_ngrams = train_phrase_model(txtfiles)
    else:
        # If bigram files already exist, then load the bigram model (if it already exists) or otherwise train a new one
        trigram_mod, bigram_mod, valid_ngrams = get_phrase_model(txtfiles)

    # Generate the trigram files using the phrase model
    bigram_files = UT.get_files_from_dir(CONFIG.BIGRAM_DIR, file_ext=".pkl")
    print("Process bigram files...")
    results = gen_phrased_sents_from_bigram_pkl_files_mp(bigram_files, trigram_mod, valid_ngrams,
                                                         workers=workers)
    for sents, pkl_file in results:
        trigram_file_name = os.path.join(CONFIG.TRIGRAM_DIR, os.path.basename(pkl_file))
        UT.save_pickle(sents, trigram_file_name)
        # print(trigram_file_name, len(sents))

# Generate cleaned sents for w2v training by reading preprocessed files from the "trigram" folder
def gen_cleaned_sents_for_w2v(txtfiles, workers=CONFIG.WORKERS):
    # First generate, then save to disk. If already exits, the program will not generate
    gen_phrased_sents_and_save_to_disk(txtfiles, workers=workers)
    # Then load from the disk (this is fast, mp not used)
    sentences = UT.Generator2Iterator(UT.load_sents_from_pkl_files, paras=CONFIG.TRIGRAM_DIR)
    return sentences  #An iterator of tokenized, lemmatized, phrased sents


if __name__ == "__main__":
    source_dir = r"D:\sample_proxy_statements"
    txtfiles = UT.get_files_from_dir(source_dir)
    sents = gen_cleaned_sents_for_w2v(txtfiles)
    # To see how the preprocessed inputs look like
    for s in sents:
        print("*"*100)
        print(s)







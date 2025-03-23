import numpy as np
import pandas as pd
import logging
from gensim.models import Word2Vec
import os

import JIS_HC0_config as CONFIG
import JIS_HC1_util as UT
from JIS_HC5_preprocess_for_word2vec import gen_cleaned_sents_for_w2v


# Train a word2vec model
def train_w2v_model(sents):
    w2v_model = Word2Vec(
        sents,
        vector_size = CONFIG.w2v_VECTOR_SIZE,
        window = CONFIG.w2v_WINDOW,
        min_count = CONFIG.w2v_MIN_COUNT,
        workers = CONFIG.w2v_WORKERS,
        epochs = CONFIG.w2v_EPOCHS
    )
    w2v_model.save(os.path.join(CONFIG.W2V_FOLDER, "w2v.model"))
    return w2v_model

# Load seed words from the CSV file
def load_seedwords_from_csv(csv_file = CONFIG.SEEDWORD_CSV):
    df = pd.read_csv(csv_file)
    df = df.fillna('')
    def clean_word_list(words):
        wl = []
        for w in words:
            w = w.strip()
            if len(w) == 0:
                continue
            w = w.lower()
            w = w.split(" ")
            w = "_".join(filter(bool,w)) # Join phrases with "_"
            wl.append(w)
        return wl
    seedword_dict = df.to_dict('list')
    seedword_dict = {k: clean_word_list(v) for k, v in seedword_dict.items()}
    return seedword_dict

# Get the most similar words for a lookup word
def get_most_similar_words(
        w2v_model, lookup_word, topn=CONFIG.TOPN, min_score=CONFIG.MIN_SCORE, absolute_score=CONFIG.ABSOLUTE
):
    # If "TOPN" is specified:
    if topn and isinstance(topn, int):
        # This is a list of tuples [('equality', 0.9491965770721436), ...]
        top_words = w2v_model.wv.most_similar(positive=[lookup_word], topn=topn)
        # Also get the frequency of the word in the corpus
        top_words = [[word, score, w2v_model.wv.get_vecattr(word,"count")] for word, score in top_words]
    elif isinstance(min_score, int) or isinstance(min_score, float):
        # This returns all the words in the vocabulary
        voca_len = len(w2v_model.wv)
        all_similar_words = w2v_model.wv.most_similar(positive=[lookup_word], topn=voca_len)
        # Then screen the words based on their cosine similarity scores with the lookup word
        if absolute_score:
            # Based on the absolute value of the score
            top_words = [[word, score, w2v_model.wv.get_vecattr(word,"count")] for word, score in all_similar_words
                         if abs(score) >= min_score]
        else:
            top_words = [[word, score, w2v_model.wv.get_vecattr(word, "count")] for word, score in all_similar_words
                         if score >= min_score]
    else:
        raise Exception("Please specify TOPN or min-score")
    df = pd.DataFrame(top_words, columns=["SimWord", "SimScore", "SimWordFreq"])
    return df

# Get the most similar words for all the seedwords and return them as a data frame
def get_most_similar_words_for_seedwords(
        w2v_model, topn=CONFIG.TOPN, min_score=CONFIG.MIN_SCORE, absolute_score=CONFIG.ABSOLUTE
):
    seedwords_dict = load_seedwords_from_csv()
    df0 = pd.DataFrame()
    for cat in seedwords_dict.keys():
        wl = seedwords_dict.get(cat)
        for seedword in wl:
            if seedword in w2v_model.wv.key_to_index:
                # Also get the frequency of the seedword in the corpus
                seed_freq = w2v_model.wv.get_vecattr(seedword, "count")
                df = get_most_similar_words(w2v_model, seedword, topn=topn, min_score=min_score, absolute_score=absolute_score)
                df["Category"] = cat
                df["SeedWord"] = seedword
                df["SeedWordFreq"] = seed_freq
            else:
                # If the seedword never appears in the corpus, then set values of relevant variables to missing
                seed_freq = 0
                term_dict = {
                    "Category": [cat],
                    "SeedWord": [seedword],
                    "SeedWordFreq": [seed_freq],
                    "SimWord": [np.NAN],
                    "SimScore": [np.NAN],
                    "SimWordFreq" : [np.NAN]
                }
                df = pd.DataFrame(term_dict)
            df0 = pd.concat([df0, df])
    df0 = df0[["Category", "SeedWord", "SeedWordFreq", "SimWord", "SimScore", "SimWordFreq"]]
    return df0

def reassign_simmilar_words(df):
    # If a word belongs to multiple categories, it is assigned to the category for which the word has the highest
    # average absolute similarity score with all the seed words in the category
    df["SimScore"] = df.SimScore.astype(float)
    df["SimScore_ABS"] = df.SimScore.abs()
    df['AvgCatSim_ABS'] = df.groupby(["SimWord", "Category"])["SimScore_ABS"].transform('mean')
    df = df.sort_values(by=["SimWord", "AvgCatSim_ABS", "SimScore_ABS"], ascending=[True, False, False])
    # If a word is matched with multiple seeds, keep the pair that has the highest AvgCatSim_ABS and SimScore_ABS.
    df = df.drop_duplicates(subset=["SimWord"], keep="first")
    df = df.sort_values(by=["Category", "SeedWord", "SimWord"])
    return df

# Train a word2vec model using the txtfiles as inputs, and then output the most similar words
def train_w2v_model_and_get_similar_words(txtfiles):
    try:
        w2v_model = Word2Vec.load(os.path.join(CONFIG.W2V_FOLDER,"w2v.model"))
        print("w2v_model already exists.")
        print("To train a new model, remove the 'w2v.model' file in the 'word2vec' folder and re-run the program")
    except:
        print("No w2v model found. Start to train a new model...")
        sentences = gen_cleaned_sents_for_w2v(txtfiles)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        w2v_model = train_w2v_model(sentences)

    # Generating and outputting a CSV file of most similar words
    df = get_most_similar_words_for_seedwords(w2v_model)
    # The following CSV file can be huge for a large corpus - commented out;
    # If you want to output the file, uncomment it
    # df.to_csv(os.path.join(CONFIG.W2V_FOLDER, "similar_words.csv"), index=False)
    df = reassign_simmilar_words(df)
    # This CSV file is much smaller with the duplicates dropped
    df.to_csv(os.path.join(CONFIG.W2V_FOLDER, "similar_words_reassigned.csv"), index=False)
    print("CSV file of most similar words generated and saved to", CONFIG.W2V_FOLDER)


if __name__ == "__main__":
    source_dir = r"D:\sample_proxy_statements"
    txtfiles = UT.get_files_from_dir(source_dir)
    train_w2v_model_and_get_similar_words(txtfiles)






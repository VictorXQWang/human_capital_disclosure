import os
from multiprocessing import Pool
from multiprocessing import cpu_count
import tqdm

cpu_cores = cpu_count()

import JIS_HC0_config as CONFIG
import JIS_HC1_util as UT
from JIS_HC2_load_keywords import keywords, ngrams, wordlist_dict

#Step 1: Load the keyword list from the CSV file

cat = [
    "Diversity, Equity, and Inclusion (DEI)", "Health and Safety (General)", "Health and Safety (COVID)", "Labor relations and culture",
    "Compensation and Benefits", "Demographics and others"
]

# Get tokenized, lemmatized sentences from the txtfile and join phrases
def get_sent_tokens_from_txtfile(txtfile, lemmatizer=CONFIG.LEMM):
    lemm_sents = UT.get_tokenized_lemmatized_sents_from_txtfile(txtfile, lemmatizer=lemmatizer)
    # Join phrases
    lemm_ngram_sents = UT.join_phrases(lemm_sents, ngrams.copy())
    return lemm_ngram_sents
    # Return a list of tokenized sentences; each sentence is a list of tokens that are lemmatized and phrased

# Get keyword counts in the sentences
def get_keyword_counts(tokenized_sents, wl_dict=wordlist_dict.copy()):
    all_tokens = UT.flatten_list_of_lists(tokenized_sents)
    wc_dict = {c: 0 for c in cat}
    for c, wl in wl_dict.items():
        for t in all_tokens:
            if t in wl:
                wc_dict[c] += 1
    wc_dict["Total"] = sum(wc_dict.values())
    return wc_dict

# Get sentences that contain any of the keywords
def get_keyword_sents(tokenized_sents, keywords=keywords.copy()):
    kw_sents = [s for s in tokenized_sents if any(kw in s for kw in keywords)]
    return kw_sents

def get_word_count(tokens, puncs_set = UT.puncs_set):
    # Do not count punctuations as words or "'s" as words ("company's" may have been tokenized into "company" and "'s")
    # The word count obtained in this way approximates that from MS-Word
    tokens = [t for t in tokens if t not in puncs_set and t not in {"'s", "'S"}]
    return len(tokens)

# Get the number of sentences containing keywords and the word count of such sentences
def get_kw_sent_counts(tokenized_sents):
    kw_sents = get_keyword_sents(tokenized_sents)
    kw_sc = len(kw_sents)
    kw_sent_wc = get_word_count(UT.flatten_list_of_lists(kw_sents))
    return kw_sc, kw_sent_wc

# Generate the HC variables from a single txt file
def gen_HC_variables_from_txtfile(txtfile):
    lemm_ngram_sents = get_sent_tokens_from_txtfile(txtfile)
    wc_dict = get_keyword_counts(lemm_ngram_sents)
    kw_sc, kw_sent_wc = get_kw_sent_counts(lemm_ngram_sents)
    wc_doc = get_word_count(UT.flatten_list_of_lists(lemm_ngram_sents))
    sc_doc = len(lemm_ngram_sents)
    data = list(wc_dict.values()) + [kw_sc, kw_sent_wc, sc_doc, wc_doc]
    data = [txtfile] + data
    return data

# Generate the HC variables for all text files in a folder using multiple workers
def gen_HC_variables_from_txtfiles(source_dir, out_csv_file, workers=CONFIG.WORKERS, chunksize=10):

    txtfiles = UT.get_files_from_dir(source_dir)
    print("Number of files to process: ", len(txtfiles), UT.current_timestamp())
    log_header = [
        "FileName", "WC_DEI", "WC_Health&Safety_General", "WC_Health&Safety_COVID",
        "WC_LaborRelations&Culture", "WC_Compensations&Benefits", "WC_Demo&Others", "KW_Count_Total",
        "KW_SC", "KW_Sent_WC", "SC_DOC", "WC_DOC"
                  ]
    if not os.path.isfile(out_csv_file):
        UT.write_results_to_csv(out_csv_file, log_header)

    pool = Pool(processes=workers)
    log_rows = tqdm.tqdm(pool.imap_unordered(gen_HC_variables_from_txtfile, txtfiles, chunksize=chunksize), total=len(txtfiles))
    for log_row in log_rows:
        UT.write_results_to_csv(out_csv_file, log_row)



if __name__ == "__main__":
    source_dir = r"D:\sample_proxy_statements"
    out_csv = r"proxy_HC_data.csv"
    print("Start to generate HC variables on", UT.current_timestamp())
    gen_HC_variables_from_txtfiles(source_dir, out_csv, workers=10)
    print("Done!", UT.current_timestamp())















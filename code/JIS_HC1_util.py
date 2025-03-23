import os
import csv
import string
import pickle
import datetime
import re

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy.tokens import Doc
import pandas as pd

import JIS_HC0_config as CONFIG

nlp = spacy.load("en_core_web_sm")
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Get current time stamp
def current_timestamp():
    return datetime.datetime.now()

# Load seed words from the csv file
def get_seedwords():
    df = pd.read_csv(CONFIG.SEEDWORD_CSV)
    df = df.fillna("")
    seedwords = []
    for col in df.columns:
        seedwords += df[col].to_list()
    seedwords = [w.strip() for w in seedwords if w.strip()]
    return seedwords

# Get seedwords that are phrases
def get_phrases_from_seedwords():
    seedwords = get_seedwords()
    phrases_seed = [w for w in seedwords if " " in w]
    phrases_seed = [w.replace(" ", "_") for w in phrases_seed]
    return phrases_seed

# Convert seed words in the source text to their most common forms
# You can customize this function according to your seed word list
def normalize_seedwords_in_text(text):
    text = re.sub(r"\bCOVID\s*19\b", "COVID-19", text, flags=re.I)
    text = re.sub(r"\bwellbeing\b", "well-being", text, flags=re.I)
    text = re.sub(r"\bhead\s+count[s]?\b", "headcount", text, flags=re.I)
    text = re.sub(r"\bwork\s+force[s]?\b", "workforce", text, flags=re.I)
    text = re.sub(r"\bwhite\s+collar\b", "white-collar", text,flags=re.I)
    return text

# Generate the list of files in the directory
def get_files_from_dir(source_dir, file_ext = ".txt", all_files = False):
    # can pass in a string or a list as file extensions

    l_files = []
    for path, dirs, files in os.walk(source_dir):
        for filename in files:
            if all_files:
                l_files.append(os.path.join(path, filename))
            else:
                if isinstance(file_ext, list):
                    if any(filename.endswith(e) for e in file_ext):
                        l_files.append(os.path.join(path, filename))
                else:
                    if filename.endswith(file_ext):
                        l_files.append(os.path.join(path, filename))
    return sorted(l_files)

# Read the text from a txt file
def get_text_from_file(txtfilename):
    with open(txtfilename, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        return text

# Remove non-ASCII characters in the string
def remove_non_ascii_chars(s):
    return "".join(i for i in s if ord(i)<128)

# Convert a list of lists into a flat list
def flatten_list_of_lists(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

# Generate a set of punctuations, excluding "-" and "_"
def gen_puncs():
    puncs = string.punctuation
    puncs = puncs.replace("-", "")
    puncs = puncs.replace("_", "")
    puncs_set = set(puncs)
    return puncs_set

puncs_set = gen_puncs()

# You can customize the following function to use your preferred tokenizer
# It currently uses the NLTK sentence tokenizer
def sent_tokenizer(text):
    sents = sent_tokenize(text)
    return sents

# Take a list of tokenized sentences (each of which is a list of tokens)
# and join tokens into phrases using "_" based on the list of phrases specified in "ngrams"
def join_phrases(tokenzied_sents, ngrams):
    sents_str = [" ".join(sent) for sent in tokenzied_sents]
    text = "\n".join(sents_str)
    text = " " + text + " "
    re_ngrams = [" " + a.replace("_", " ") + " " for a in ngrams] # e.g. " human resource "
    ngrams_underline = [" "  + a + " " for a in ngrams] # e.g. " human_resource "
    pattern_zip = zip(re_ngrams, ngrams_underline)
    for p in pattern_zip:
        text = text.replace(p[0], p[1]) # e.g., replace "human resource" with "human_resource"
    sents = text.split("\n")
    sents = [s.split(" ") for s in sents]
    return sents

# Append one row to the CSV file
def write_results_to_csv(csvfile, row):
    with open(csvfile, 'a',encoding='utf-8',newline = '') as csv_log:
        logwriter = csv.writer(csv_log, delimiter=',')
        logwriter.writerow(row)

# Lemmatize tokenized sentences using the WordNet Lemmatizer available with NTLK
def lemm_tokens_nltk(tokenized_sents):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    tokenized_lemm_sents = []
    for tokenized_sent in tokenized_sents:
        word_pos = pos_tag(tokenized_sent)
        lemm_words = [wnl.lemmatize(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]
        tokenized_lemm_sents.append(lemm_words)
    return tokenized_lemm_sents

# Lemmatize tokenized sentences using the "Spacy" lemmatizer
def lemm_tokens_spacy(tokenized_sents):
    # Sents are already tokenized.
    # Don't need to use the spacy sentence tokenizer, which does not perform well
    def custom_tokenizer(text):
        tokens = text.split(" ")
        return Doc(nlp.vocab, tokens)

    nlp.tokenizer = custom_tokenizer
    # spacy takes strings
    sents = [" ".join(s) for s in tokenized_sents]
    tokenized_lemm_sents = []
    for sent in nlp.pipe(sents, disable=["parser", "ner"]):
        lemm_words = [tok.lemma_ for tok in sent]
        tokenized_lemm_sents.append(lemm_words)
    return tokenized_lemm_sents

# Generate tokenized and lemmatized sentences from the txt file
def get_tokenized_lemmatized_sents_from_txtfile(txtfile, lemmatizer=CONFIG.LEMM):
    text = get_text_from_file(txtfile)
    text = remove_non_ascii_chars(text)
    text = normalize_seedwords_in_text(text)
    sents = sent_tokenizer(text)

    # Use the NLTK word tokenizer
    tokenized_sents = [word_tokenize(s) for s in sents]
    # To lower cases so that proper nouns are handled properly
    tokenized_sents = [[t.lower() for t in s] for s in tokenized_sents]
    if lemmatizer == "NLTK":
        lemm_sents = lemm_tokens_nltk(tokenized_sents)
    elif lemmatizer == "SPACY":
        lemm_sents = lemm_tokens_spacy(tokenized_sents)
    else:
        raise Exception("Wrong lemmatizer specified. Please use NLTK or SPACY.")
    return lemm_sents

# Check if a word contains any digit
def contains_digit(word):
    return any(map(str.isdigit, word))

# Check if a word consists of all digits
def contains_all_digits(word):
    word = word.replace(",", "")
    word = word.replace(":", "")
    word = word.replace(".", "")
    return all(map(str.isdigit, word))

# Check if a word contains any punctuation (excluding "-" and "_")
def contains_punc(word):
    return any(char in puncs_set for char in word)

# Check if a word is a stopword; the word must be first converted to lower case
def is_stopword(w):
    return w in stop_words

# Save an object to the disk in binary
def save_pickle(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

# Load an object from a pickle file in binary
def load_pickle(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data

# Save preprocessed sentences to a binary file onto the disk
def save_sents_to_pkl_file(sentences, target_dir, chunksize=CONFIG.NUM_SENTS):
    # sentences can be a generator, and the length is not readily available
    print("Starting to save preprocessed sentences to disk")
    sents_to_write = []
    for i, sent in enumerate(sentences):
        sents_to_write.append(sent)
        if (i+1) % chunksize == 0:
            outfile = os.path.join(target_dir,f"sents_list_{str(i+1)}.pkl")
            save_pickle(sents_to_write, outfile)
            sents_to_write = []
    # Output the partial chunk
    outfile = os.path.join(target_dir, "sents_list_last.pkl")
    save_pickle(sents_to_write, outfile)
    # print("Sents saved to ", outfile)

# Load sents from the saved binary file and output as a generator
def load_sents_from_pkl_files(target_dir):
    files = get_files_from_dir(target_dir, file_ext=".pkl")
    for f in files:
        sents = load_pickle(f)
        for s in sents:
            yield s

# You can customize the white list of valid words
# Typically words containing numbers or punctuations are excluded
white_list_word = {
"401K", "401-K", "401(K)", "COVID-19", "LGBT+", "LGBTQ+", "LGBTIQ+", "LGBTIQA+", "LGBTQ2",  "LGBT2Q+"
}

white_list_word = set([t.lower() for t in white_list_word])

# Check if a word is a valid word
def is_valid_word(w):

    # Bypass those on the whitelist
    if w in white_list_word:
        return True

    if len(w)<2:  #Remove single-letter words
        return False
    if w.startswith("-") or w.startswith("_"):
        return False
    if is_stopword(w): # Remove stopword
        return False

    if contains_digit(w): # Remove words containing digits
        return False
    if contains_punc(w): # Remove words containing punctuations (excluding "_" and "-")
        return False
    return True

# You can customize the whitelist of phrases
white_list_ngram = get_phrases_from_seedwords()
white_list_ngram = set([t.lower() for t in white_list_ngram])

# Discard a ngram if any one of its components is not a valid word
def is_valid_ngram(ngram):
    # Bypass those on the whitelist
    if ngram in white_list_ngram:
        return True

    tokens=ngram.split("_")
    for t in tokens:
        if not is_valid_word(t): # dropped of a stopword
            return False
    return True

# Reduce the size of the vocabulary for training phrase by converting tokens containing digits to "1" and
# and tokens containing punctuations to "," (excluding those on the whitelist)

def normalize_token(w):
    #Bypass the words on the whitelist
    if w in white_list_word:
        return w

    if w in white_list_ngram:
        return w

    if contains_digit(w):
        return "1"
    elif contains_punc(w):
        return ","
    else:
        return w

# Convert a generator into an iterator for model training
# Source: https://stackoverflow.com/questions/56468865/sentence-iterator-to-pass-to-gensim-language-model
# To use multiple processing, this class has to be kept in this module.

class Generator2Iterator():
    # pass in the function and/or arguments of the function
    def __init__(self, generator_function, paras=None):
        self.generator_function = generator_function
        self.paras = paras
        if paras:
            self.generator = self.generator_function(self.paras)
        else:
            self.generator = self.generator_function()

    def __iter__(self):
        # reset the generator
        if self.paras:
            self.generator = self.generator_function(self.paras)
        else:
            self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result

# Write text to a txt file
def write_text_to_txtfile(filename, text, flag = 'w'):
    with open(filename, flag, encoding='utf-8', errors='ignore') as outfile:
        outfile.write(text)
    return

# Function for parsing the HC disclosures data in "JIS_Data_HC_Disclosures.txt"
# Please specify the path of this file as well as the folder for the outputs
def parse_hcm_disclosure_text(data_file, to_txtfiles=True, outdir=None, to_csv_file=True):
    text = get_text_from_file(data_file)
    docs = re.findall(r"<DOC\d+>.+?</DOC\d+>", text, flags=re.DOTALL)
    if outdir:
        os.chdir(outdir)
    table = []
    for doc in docs:
        disc_text = re.search(r"<text>(.+?)</text>", doc, flags=re.DOTALL).group(1)
        disc_text = disc_text.strip()
        ids = re.search(r"<(CIK:.+?)>", doc).group(1)
        cik, comp, fdate, reportperiod  = [e.split(":")[1].strip() for e in ids.split(";")]
        filename = "_".join([cik, comp, fdate, reportperiod]) + ".txt"
        if to_txtfiles:
            write_text_to_txtfile(filename, disc_text)
        if to_csv_file:
            row = [cik, comp, fdate, reportperiod, disc_text]
            table.append(row)
    if to_csv_file:
        df = pd.DataFrame(table, columns=["CIK", "Company", "Fdate", "ReportPeriod", "HC"])
        df.to_csv("HC disclosures.csv", index=False)


if __name__=="__main__":
    # to Parse the "JIS_Data_HC_Disclosures.txt"
    data_file = r"D:\JIS_HC_Project\Data\JIS_Data_HC_Disclosures.txt"
    out_dir = r"D:\JIS_HC_Project\Data\HCD"
    parse_hcm_disclosure_text(data_file, to_txtfiles=True, outdir=out_dir, to_csv_file=True)

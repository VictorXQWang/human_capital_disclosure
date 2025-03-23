import pandas as pd
import JIS_HC0_config as CONFIG
import JIS_HC1_util as UT

# Load the keywords from the seedword CSV file
def load_keyword_list(wl_csv):
    df = pd.read_csv(wl_csv)
    # convert to a list
    def to_list(r):
        l = r.split(",")
        l = [e.strip() for e in l]
        l = sorted(l, key=len, reverse=True)
        return l

    df["WordList"] = df.WordList.apply(to_list)

    # Get all the keywords:
    keywords = set(UT.flatten_list_of_lists(df.WordList))
    # Get all Ngrams from the keywords
    ngrams = [w for w in keywords if "_" in w]
    ngrams = sorted(ngrams, key=len, reverse=True)
    # Generate a wordlist dictionary
    wordlist_dict = dict(df[["Category", "WordList"]].values)
    wordlist_dict = {k: set(v) for k, v in wordlist_dict.items()}
    return keywords, ngrams, wordlist_dict

keywords, ngrams, wordlist_dict = load_keyword_list(CONFIG.HC_WL_CSV)

if __name__ == "__main__":
    keywords, ngrams, wordlist_dict = load_keyword_list(CONFIG.HC_WL_CSV)
    print(keywords)
    print(ngrams)
    print(wordlist_dict)

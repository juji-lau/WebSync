from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scratch

def main():
    webnovels = scratch.get_webnovel_data()
    webnovels_tokenized = scratch.tokenize_webnovels(scratch.tokenize, webnovels)
    print(webnovels_tokenized)
    # Seems to be a list of dictionaries that are Dict[index : , tokenized_description :]

    webnovels_descs = []
    for webtok_dict in webnovels:
        webnovels_descs.append(webtok_dict['description'])

    print("")
    print("")
    print(webnovels_descs)
    #webnovels_descs = list(webnovels_tokenized.values())

    fanfics = scratch.get_fanfic_data()
    #fanfics_tokenized = scratch.tokenize_fanfics(scratch.tokenize, fanfics)
    
    fanfic_descs = []
    for fanfic_dict in fanfics:
        fanfic_descs.append(fanfic_dict['description'])


    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .95, min_df = .10)

    webnovels_tfidf_matrix = vectorizer.fit_transform(webnovels_descs)
    print(webnovels_tfidf_matrix.shape)
    fanfic_tfidf_matrix = vectorizer.fit_transform(fanfic_descs)
    print(fanfic_tfidf_matrix.shape)


    u_web, s_web, v_trans_web = svds(webnovels_tfidf_matrix, k=100)
    u_fic, s_fic, v_trans_fic = svds(fanfic_tfidf_matrix, k=100)

    # %matplotlib inline
    plt.xlabel("Singular value number")
    plt.ylabel("Singular value")
    plt.plot(s_web[::-1])
    plt.savefig("s_web.png")
    plt.show()

    plt.plot(s_fic[::-1])
    plt.xlabel("Singular value number")
    plt.ylabel("Singular value")
    plt.savefig("s_fic.png") 
    plt.show()

    #n_fanfics = len(fanfics)

if __name__ == "__main__":
    main()
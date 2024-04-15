import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scratch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Decided not to use SVD to convert between webnovel to fanfic descriptions
# It did not produce better results when compared with cosine similarity

def main():
    webnovels = scratch.get_webnovel_data()
    fanfics = scratch.get_fanfic_data()

    webnovels_descs = []    # A list of description strings
    for webtok_dict in webnovels:
        webnovels_descs.append(webtok_dict['description'])
    
    fanfic_descs = []
    for fanfic_dict in fanfics:
        fanfic_descs.append(fanfic_dict['description'])

    combined_descs = webnovels_descs + fanfic_descs

    vectorizer =  TfidfVectorizer(stop_words = 'english')
    vectorizer.fit(combined_descs)
    webnovels_tfidf_matrix = vectorizer.transform(webnovels_descs)
    fanfics_tfidf_matrix = vectorizer.transform(fanfic_descs)
    combined_tfidf_matrix = vectorizer.transform(combined_descs)

    print(webnovels_tfidf_matrix.shape)
    print(fanfics_tfidf_matrix.shape)

    # Used when doing both fit and transform in one step
    # webnovels_tfidf_matrix = vectorizer.fit_transform(webnovels_descs)
    # print(webnovels_tfidf_matrix.shape)
    # fanfic_tfidf_matrix = vectorizer.fit_transform(fanfic_descs)
    # print(fanfic_tfidf_matrix.shape)

    # When using the linear algebra approach of SVD
    # u_webdocs_compressed, s_web, vt_webwords_compressed = svds(webnovels_tfidf_matrix, k=150)
    # webwords_compressed = vt_webwords_compressed.transpose()
    # u_ficdocs_compressed, s_fic, vt_ficwords_compressed = svds(fanfics_tfidf_matrix, k=150)

    svd = TruncatedSVD(n_components=150)
    svd.fit(combined_tfidf_matrix)
    u_webdocs_compressed = svd.transform(webnovels_tfidf_matrix)
    u_fanfics_compressed = svd.transform(fanfics_tfidf_matrix)

    # Used to print out graphs to select the number of n_componets
    # plt.title("Webnovels Descriptions")
    # plt.xlabel("Singular value number")
    # plt.ylabel("Singular value")
    # plt.plot(s_web[::-1])
    # plt.savefig("s_web.png")
    # plt.show()

    # plt.title("Fanfics Descriptions")
    # plt.xlabel("Singular value number")
    # plt.ylabel("Singular value")
    # plt.plot(s_fic[::-1])
    # plt.savefig("s_fic.png") 
    # plt.show()
    
    print("Shapes")
    print(webnovels_tfidf_matrix[0].toarray().shape)
    print(fanfics_tfidf_matrix.toarray().shape)
    print("")

    sims = cosine_similarity(webnovels_tfidf_matrix[0].toarray(), fanfics_tfidf_matrix.toarray()).flatten()
    result_indices = np.argsort(sims)[-1]
    print("Comparing the webnovel and fanfic descriptions")
    print("Webnovel Description:")
    print(webnovels_descs[0])
    print("")
    print("Fanfic Description:")
    print(fanfic_descs[result_indices])

if __name__ == "__main__":
    main()
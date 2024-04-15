import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scratch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD



def main():
    webnovels = scratch.get_webnovel_data()
    #print(webnovels)
    #webnovels_tokenized = scratch.tokenize_webnovels(scratch.tokenize, webnovels)
    #print(webnovels_tokenized)
    # Seems to be a list of dictionaries that are Dict[index : , tokenized_description :]

    webnovels_descs = []    # A list of description strings
    for webtok_dict in webnovels:
        webnovels_descs.append(webtok_dict['description'])

    # print("")
    # print(webnovels_descs)
    # print("")

    #webnovels_descs = list(webnovels_tokenized.values())

    fanfics = scratch.get_fanfic_data()
    #fanfics_tokenized = scratch.tokenize_fanfics(scratch.tokenize, fanfics)
    
    fanfic_descs = []
    for fanfic_dict in fanfics:
        fanfic_descs.append(fanfic_dict['description'])

    print("")
    #print(fanfic_descs)
    print("")

    combined_descs = webnovels_descs + fanfic_descs

    #vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .95, min_df = .10)
    #   For some reason, it did not like my max_df and my min_df
    vectorizer =  TfidfVectorizer(stop_words = 'english')
    vectorizer.fit(combined_descs)

    webnovels_tfidf_matrix = vectorizer.transform(webnovels_descs)
    fanfics_tfidf_matrix = vectorizer.transform(fanfic_descs)
    combined_tfidf_matrix = vectorizer.transform(combined_descs)

    print(webnovels_tfidf_matrix.shape)
    print(fanfics_tfidf_matrix.shape)

    # webnovels_tfidf_matrix = vectorizer.fit_transform(webnovels_descs)
    # print(webnovels_tfidf_matrix.shape)
    # fanfic_tfidf_matrix = vectorizer.fit_transform(fanfic_descs)
    # print(fanfic_tfidf_matrix.shape)


    # u_webdocs_compressed, s_web, vt_webwords_compressed = svds(webnovels_tfidf_matrix, k=150)
    # webwords_compressed = vt_webwords_compressed.transpose()

    # u_ficdocs_compressed, s_fic, vt_ficwords_compressed = svds(fanfics_tfidf_matrix, k=150)

    # print("Compressed shapes")
    # print(u_webdocs_compressed.shape)
    # print(u_ficdocs_compressed.shape)

    svd = TruncatedSVD(n_components=150)
    svd.fit(combined_tfidf_matrix)
    u_webdocs_compressed = svd.transform(webnovels_tfidf_matrix)
    u_fanfics_compressed = svd.transform(fanfics_tfidf_matrix)

    


    # %matplotlib inline
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

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    # words_compressed_normed = normalize(webwords_compressed, axis = 1)
    # td_matrix_np = webnovels_tfidf_matrix.transpose().toarray()
    # td_matrix_np = normalize(td_matrix_np)

    # cosine similarity
    # def closest_words(word_in, words_representation_in, k = 10):
    #     if word_in not in word_to_index: return "Not in vocab."
    #     sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    #     asort = np.argsort(-sims)[:k+1]
    #     return [(index_to_word[i],sims[i]) for i in asort[1:]]

    # word = 'male'
    # print("Using SVD:")
    # for w, sim in closest_words(word, words_compressed_normed):
    #     try:
    #         print("{}, {:.3f}".format(w, sim))
    #     except:
    #         print("word not found")
    # print()



    print("")
    docs_compressed_normed = normalize(u_webdocs_compressed)

    def closest_projects(project_index_in, project_repr_in, k = 5):
        sims = project_repr_in.dot(project_repr_in[project_index_in,:])
        asort = np.argsort(-sims)[:k+1]
        return [(fanfic_descs[i],sims[i]) for i in asort[1:]]
    
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
    

    # for i in range(10):
    #     print("INPUT PROJECT: "+webnovels_descs[i])
    #     print("CLOSEST PROJECTS:")
    #     print("Using SVD:")
    #     for title, score in closest_projects(i, docs_compressed_normed):
    #         print("{}:{:.3f}".format(title, score))
    #     print()
    #     print("Not using SVD (using directly term-document matrix):")
    #     for title, score in closest_projects(i, td_matrix_np):
    #         print("{}:{:.3f}".format(title, score))
    # print("--------------------------------------------------------\n")



if __name__ == "__main__":
    main()
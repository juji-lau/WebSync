"""
Name: analysis.py
Author: Juji Lau
Description: 
This file stores the functions for ranking and retrieving documents by relevance.

Edit distance: For finding the webnovel title
SVD analysis: For finding the webnovel by description
Cosine Similarity: For ranking the fanfictions by similarity to each webnovel.
"""
import numpy as np
from typing import Callable
import math
from collections import Counter
import json
import re 
from tqdm import tqdm
from matplotlib import pyplot as plt
import logger

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words, TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from preprocess import (
    get_fanfic_data, 
    get_webnovel_data, 
    tokenize_fanfics, 
    tokenize_webnovels,
    tokenize
)

# Set up logging for the file
logging = logger.get_logger()

""" ------------------------------ FUCNTIONS TO RANKING ------------------------------------ """
def build_inverted_index(msgs: list[dict]) -> dict:
    """Builds an inverted index from either webnovels or fanfics.

    Arguments
    =========
    msgs: list[dict{id:int, description:str}]
        Each message in this list already has a 'tokenized_description'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_index, count_of_term_in_doc)
        such that tuples with smaller doc_indexes appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'tokenized_description': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'tokenized_description': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    iid = {}
    for msg in range(0, len(msgs)):
        toks = msgs[msg]['tokenized_description']
        first_inst = {} # dictionary for each doc of first instances
        for tok in toks:
            #first_inst = false by default
            # token is not in dictionary
            if tok not in iid:
                tup = (msg, 1)
                iid[tok] = [(tup)]
                first_inst[tok] = 0
            else:       # token is in the dictionary
                # First instance for this doc
                if tok not in first_inst:
                    tup = (msg, 1)
                    iid[tok].append(tup)
                    first_inst[tok] = len(iid[tok])-1
                else:
                    # increment count
                    curr_count = (iid[tok])[first_inst[tok]][1]
                    tup = (msg, curr_count+1)
                    (iid[tok])[first_inst[tok]] = tup
    return iid

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Arguments
    =========
    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======
    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = {}
    for term in inv_idx: 
        term_df = len(inv_idx[term])

        if term_df/n_docs <= max_df_ratio and term_df >= min_df:
            idf[term] = math.log2(n_docs/(1 + term_df))
    return idf

def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    Arguments
    =========
    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    norms = np.zeros((n_docs))

    for term in idf: 
        score = idf[term]
        term_docs = index[term]
        for doc, tf in term_docs: 
            norms[doc] += (tf*score)**2
    return np.sqrt(norms)

def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========
    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.

    Returns 
    =========
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_scores = {}
    influential_words = {}
    for query_word in query_word_counts: 
        query_word_count = query_word_counts[query_word]
        for (doc, tf) in index[query_word]:
            score = tf*idf[query_word]*query_word_count*idf[query_word]
            if doc not in doc_scores: 
                doc_scores[doc] = score
                influential_words[doc] = [(query_word, score)]
            else:
                doc_scores[doc] += score
                influential_words[doc].append((query_word, score))

    for doc in influential_words: 
        influential_words[doc] = sorted(influential_words[doc], key=lambda x: x[1], reverse=True)[:5]
        influential_words[doc] = [t[0] for t in influential_words[doc]]

    return doc_scores, influential_words

def compute_cossim_for_webnovel(
    webnovel_toks: dict[str, str],
    fanfic_inv_index: dict,
    fanfic_idf,
    fanfic_norms,
    score_func = accumulate_dot_scores,
) -> list[tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    webnovel_toks: string,
        The webnovel.

    fanfic_inv_index: an inverted index as above

    fanfic_idf: idf values precomputed as above

    fanfic_norms: fanfic norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See accumulate_dot_scores)

    Returns
    =======

    results, list of tuples (score, fanfic_index)
        Sorted list of results such that the first element has
        the highest score, and `fanfic_index` points to the fanfic
        with the highest score. Only the first ten. 

    """
    webnovel_unique_toks = list(set(webnovel_toks))
    webnovel_word_counts = dict(Counter(webnovel_unique_toks)) 

    norm = 0
    for term in webnovel_unique_toks: 
        if term in fanfic_idf and term in webnovel_word_counts:
            norm += (webnovel_word_counts[term]*fanfic_idf[term])**2
        elif term not in fanfic_idf and term in webnovel_word_counts:
            del webnovel_word_counts[term]
    norms = math.sqrt(norm) * fanfic_norms

    doc_scores, influence_words = score_func(webnovel_word_counts, fanfic_inv_index, fanfic_idf)

    cossim = []
    for doc in doc_scores:
        cossim.append((doc_scores[doc]/norms[doc], doc, influence_words[doc]))

    result = sorted(cossim, key=lambda x: x[0], reverse=True)[:50]
    return result
    

def build_sims_cos(webnovels_tokenized, fanfic_inv_index, fanfic_idf, fanfic_norms, score_func, input_get_sims_method):
    """Returns a cosine similarity dictionary with len(webnovels_tokenized) keys:
        [webnovel_index] should be the ranked list of cosine similarity between the webnovel and all fanfics 
    
    Arguments
    =========

    input_get_sim_method: function,
        a function to compute a ranked cosine similarity list between a singular webnovel and all relevant fanfics
        (look at compute_cossim_for_webnovel)

    Returns
    =========
    webnovel_sims: dict
        The key is a webnovel_index and the value is a ranked list of cosine similarity (cos sim score, fanfic_index)
    """
    webnovel_sims_and_influential_words = {} # key - webnovel id, value = list of sorted fanfics by similarity (cos sim score, fanfic index)
    n_webnovels = len(webnovels_tokenized)

    for i in tqdm(range(n_webnovels)):
        webnovel = webnovels_tokenized[i]
        results = input_get_sims_method(webnovel['tokenized_description'], fanfic_inv_index, fanfic_idf, fanfic_norms, score_func)
        webnovel_sims_and_influential_words[webnovel['index']] = results

    return webnovel_sims_and_influential_words

def get_svd_tags(webnovel_data, fanfic_data, num_tags=235, sim_threshold=0.35, min_doc_frequency=0.005, max_doc_frequency=0.95, t_pattern="%", regex_t_pattern="[^%]+"):
    """
    Retrieves only tags from the webnovel and fanfic data, then filters the tags using SVD to generate a smaller list of tags    
   
   Arguments
    =========
    webnovel_data: The webnovels dictionary
    fanfic_data: The fanfic dictionary
    num_dims: The number of tags we want the final tagset to contain
    sim_threshold: Optional argument to decide at what threshold we combine tags
    min_doc_frequency: [1, 0], min percentage of works a tag appears in
    max_doc_frequency: [1, 0], max percentage of works a tag appears in; >= min_doc_frequency
    tpattern: a string which each tag is joined by

    Returns
    =========
    tags_list: list of valid tags that the user can enter of size num_dims
    work2index: list of works, in order of the rows of the svd matrix
    tags: list of tags, in order of the columns of the svd matrix
    tf_matrix: the tfidf matrix where columns are tags, and rows are documents

    """
    # Gather all tags and associated works
    work2index = []
    # all_ tags is a list of dicts, where each dict has the title and tags
    all_tags = []
    for w in webnovel_data:
        # record the work title
        work2index.append(w["titles"][0])
        # make the list of tags into a single string
        tag_string = t_pattern.join(w["tags"])
        all_tags.append({"title": w["titles"][0], "tags" : tag_string})
    
    for f in fanfic_data:         
        # record the work title
        work2index.append(f["title"])
        # make the list of tags into a single string
        tag_string = t_pattern.join(f["tags"])
        all_tags.append({"title": f["title"], "tags" : tag_string})

    # make a tfidf matrix; dims = |works| x |tags|, where svd(i, j) = tf-idf score of tag j in work i
    vec = TfidfVectorizer(max_df = max_doc_frequency,
                            min_df = min_doc_frequency, stop_words='english', token_pattern=regex_t_pattern)
    td_matrix = vec.fit_transform([x["tags"] for x in all_tags]) 
    
    # do svd: 
    works_compressed, s, tags_compressed = svds(td_matrix, k=num_tags)
    tags_compressed = tags_compressed.T

    # map from tag to vec and back
    word_to_index = vec.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    tags_list = []
    # Get all the tags kept: 
    for i in range(0, num_tags):
        tags_list.append(index_to_word[i])
    return (tags_list, tags_compressed, works_compressed, s)

def filter_fanfics(fanfics, tags_list:list[str]):
    """ Takes in the list of all fan fictions in the database and filters it to 
    to only include fanfics with the tags present in tags_list
    
    Arguments
    =========
    fanfics: A list[dict] of fanfictions in the same form returned by get_fanfic_data()
    tags_list: The list of tags to filter using.  Can be user entered.

    Returns: 
    ==========
    filtered_fanfics: a list[dict] of fanfictions in the same form as the input.  
    For all fanfic in filtered_fanfics, fanfic["tags"] must contain only strings 
    present in tags_list.
    """
    # print("FANFICS: ", fanfics)
    filtered_fanfics = []
    tags_set = set([tag.lower() for tag in tags_list])

    # Lowercase all the tags for each fanfiction
    for fanfic in fanfics: 
        fanfic_tags_set = set([tag.lower() for tag in fanfic["tags"]])

        # print("Current user tagset: ", tags_set)
        # print("FANFIC tags: ", fanfic_tags_set)
        if len(fanfic_tags_set.intersection(tags_set)) > 0:
            filtered_fanfics.append(fanfic)

        logging.info(f"User tags: {tags_set} \n All tags: {fanfic_tags_set}")

    return filtered_fanfics


def main():
    fanfics = get_fanfic_data()
    webnovels = get_webnovel_data()
    # print("current fanfic length", len(fanfics), flush=True)
    # Svd stuff: 
    tags_list, tags_compressed, works_compressed, s = get_svd_tags(webnovels, fanfics)
    fanfics = filter_fanfics(fanfics, tags_list)

    n_fanfics = len(fanfics)

    fanfics_tokenized = tokenize_fanfics(tokenize, fanfics)
    webnovels_tokenized = tokenize_webnovels(tokenize, webnovels)

    fanfic_inverted_index = build_inverted_index(fanfics_tokenized)
    fanfic_idf = compute_idf(fanfic_inverted_index, n_fanfics)
    fanfic_norms = compute_doc_norms(fanfic_inverted_index, fanfic_idf, n_fanfics)

    # # Comment this when actually running
    # cossims_and_influential_words = build_sims_cos(webnovels_tokenized[:5], fanfic_inverted_index, fanfic_idf, fanfic_norms, accumulate_dot_scores, compute_cossim_for_webnovel)
    # file = 'test.json'

    # Uncomment this when actually running
    cossims_and_influential_words = build_sims_cos(webnovels_tokenized, fanfic_inverted_index, fanfic_idf, fanfic_norms, accumulate_dot_scores, compute_cossim_for_webnovel)
    file = 'webnovel_to_fanfic_cossim.json'

    with open(file, 'w', encoding='utf-8') as f:
        json.dump({'cossims_and_influential_words':cossims_and_influential_words, 'fic_id_to_index': fic_id_to_index, 'index_to_fanfic_id':index_to_fic_id, 'webnovel_title_to_index':webnovel_title_to_index, 'fanfic_id_to_popularity':fanfic_id_to_popularity, 'tags_list':tags_list}, f)


if __name__ == "__main__":
    main()

'''
====================================== Edit Distance Search for Frontend ===================================================
'''
    

def insertion_cost(message, j):
    return 1


def deletion_cost(query, i):
    return 1


def substitution_cost(query, message, i, j):
    if query[i - 1] == message[j - 1]:
        return 0
    else:
        return 1


def edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func):
    """Calculates the edit matrix

    Arguments
    =========

    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit matrix {(i,j): int}
    """

    m = len(query) + 1
    n = len(message) + 1

    chart = {(0, 0): 0}
    for i in range(1, m):
        chart[i, 0] = chart[i - 1, 0] + del_cost_func(query, i)
    for j in range(1, n):
        chart[0, j] = chart[0, j - 1] + ins_cost_func(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i - 1, j] + del_cost_func(query, i),
                chart[i, j - 1] + ins_cost_func(message, j),
                chart[i - 1, j - 1] + sub_cost_func(query, message, i, j),
            )
    return chart


def edit_distance(
    query: str, message: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int
) -> int:
    """Finds the edit distance between a query and a message using the edit matrix

    Arguments
    =========
    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit cost (int)
    """

    query = query.lower()     # rows
    message = message.lower() # cols

    # TODO-1.1
    edit_d_matrix = edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func)
    return edit_d_matrix[(len(query), len(message))]


def edit_distance_search(
    query: str,
    msgs: list[list],
    ins_cost_func: int,
    del_cost_func: int,
    sub_cost_func: int,
) -> list[tuple[int, str]]:

    """Edit distance search

    Arguments
    =========
    query: string,
        The query we are looking for.

    msgs: list of dicts,
        Each message in this list has a 'text' field with
        the raw document.

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns
    =======
    result: list of (score, message) tuples.
        The result list is sorted by score such that the closest match
        is the top result in the list.

    """
    result = []
    for msg in msgs:
      
      dist = edit_distance(query,msg['text'],ins_cost_func,del_cost_func,sub_cost_func)
      result.append((dist,msg['text']))
    result.sort(key=lambda tup: tup[0])
    print(result[:10])
    return result
    # TODO-1.2
    lst_of_tups = []
    
    #print(msgs[0])      # list of titles for a single webnovel
    for webnov_title_lst in msgs:
        webnovel_title = webnov_title_lst[0]
        #novel_title_ind = webnovel_title_to_index[webnov_title_lst[0]]
        #print("HERES THE INDEX", novel_title_ind)
        score = edit_distance(query, webnovel_title, ins_cost_func, del_cost_func, sub_cost_func)
        lst_of_tups.append((score, webnovel_title))

    sort_lst = sorted(lst_of_tups, key = lambda x: x[0])[:10]

    # lst_of_titles = [t[1] for t in sort_lst]
    
    # lst_dict_titles = []
    # for title in lst_of_titles:

    return [t[1] for t in sort_lst]
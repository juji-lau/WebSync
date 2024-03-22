#from src.utils.utils import timer
from typing import List, Tuple, Dict
from collections.abc import Callable
#from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#import src.data_processing.helpers as helpers
import math
from collections import defaultdict
from collections import Counter
import json
import re 

# Note: popularity score = (kudos * chapters) / hits
#TODO: currently webnovel["title"] gives a list of associated webnovel titles\
# right now, we are returning webnovel["title"][0].  Find a way to \
# incorporate all associated titles
#TODO: edit distance search for webnovels


"""
fic_id_to_index: maps the fanfiction id to a zero-based index. 
  fic_id_to_index[fanfic_id] = int
  
index_to_fic_id: maps a zero-based index to the fanfiction id.
  index_to_fic_id[int] = fanfic_id
"""
fic_id_to_index = {}
index_to_fic_id = {}
wn_title_to_index = {}
index_to_wn_title = {}


fanfics = []
webnovels = []
def get_fanfic_data():
    """"
    Gets the fanfic data in the form: 

    [
        {
            "rating": "General Audiences", 
            "hits": 3312, 
            "kudos": 153, 
            "description": "....", 
            "language": "English", 
            "title": "Those Kind of People", 
            "tags": ["No Archive Warnings Apply", ...], 
            "finished": "Finished", 
            "chapters": 1, 
            "authorName": "Psychsie", 
            "comments": 12, 
            "words": 6219, 
            "date": "2019-03-31", 
            "bookmarks": 12, 
            "id": 4571955
        }, ...
    ]
    """
    # files is a list of dictionaries.  List[Dict(fanfic_id, description)]
    files = ['fanfic_G_2019-processed-pg1.json', 'fanfic_G_2019-processed-pg2.json', 'fanfic_G_2019-processed-pg3.json']
    for file in files:
        with open(file, 'r') as f:
            fanfics = fanfics + json.load(f)

def get_webnovel_data():
    """"
    Gets the webnovel data in the form: 

    [
        {
            "titles": ["Title 1", "Title 2", ...], (List[str])
            "original_lang": "English",           (str) 
            "rank": 153,                          (int)
            "description": "....",                (str)
            "authors": ["author 1"],              (List[str])
            "genres": ["Horror", ...],            (List[str])
            "tags": ["Battle Competition", ...],  (List[str])
        }, ...
    ]
    """
    # files is a list of dictionaries.  List[Dict(fanfic_id, description)]
    files = ['novel_info.json']
    for file in files:
        with open(file, 'r') as f:
            webnovels = webnovels + json.load(f)

def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    List[str]
        A list of strings representing the words in the text.
    """
    return re.findall("[A-Za-z]+", text.lower())

# files is a list of dictionaries.  List[Dict(fanfic_id, description)]
def tokenize_fanfics(tokenize_method: Callable[[str], List[str]], 
    input_fanfics: List[Dict[str, str]], ) -> List[str]:
    """Returns a list of tokens contained in an entire list of fanfics. 
       Also builds fic_id_to_index{}.  

    Parameters
    ----------
    tokenize_method : Callable[[str], List[str]]
        A method to tokenize a string into a list of strings representing words.
    input_fanfics : List[Dict{}].  See get_fanfic_data() for more info
        A list of fanfiction dictionaries 
        (specified as a dictionary with ``id``, ``description``, and other tags specified above).
    
    Returns
    -------
    List[Dict{fanfic_id:str, tokenized_description:str}]
        A list of tokens for a single description, for the entire list of fanfics.
    """
    counter = 0
    tokenized_descriptions = []
    for fanfic_dict in input_fanfics:
      fanfic_id = fanfic_dict['id']
      fanfic_description = fanfic_dict['description']
      tokenized_descriptions.append({"id":fanfic_id, "tokenized_description":tokenize_method(fanfic_description)})
      # add to fic_id_to_index, and index_to_fic_id:
      fic_id_to_index[fanfic_id] = counter
      index_to_fic_id[counter] = fanfic_id
      counter+=1
    return tokenized_descriptions  

def tokenize_webnovels(tokenize_method: Callable[[str], List[str]], 
    input_webnovels: List[Dict[str, str]], ) -> List[str]:
    """Returns a list of tokens contained in an entire list of webnovels. 
      Creates an id mapping for webnovels, then populates: wn_title_to_index{} 
      and index_to_wn_title{}  

    Parameters
    ----------
    tokenize_method : Callable[[str], List[str]]
        A method to tokenize a string into a list of strings representing words.
    input_webnovels : List[Dict{}]  See get_webnovel_data() for more info.
        A list of webnovel dictionaries 
        (specified as a dictionary with ``title``, ``description``, and other tags specified above).

    Returns
    -------
    List[Dict{webnovel_id : int, tokenized_descriptions:str}]
        A list of dictionaries, where key == webnovel_id, and value == tokenized description
        Note: the webnovel_id is an index, which can be mapped to the *first* title of the associated webnovel.
    """
    counter = 0
    tokenized_descriptions = []
    for webnovel_dict in input_webnovels:
      webnovel_title = webnovel_dict['titles'][0]
      webnovel_description = webnovel_dict['description']
      tokenized_descriptions.append({"id":counter, "tokenized_description":tokenize_method(webnovel_description)})
      # add to fic_id_to_index, and index_to_fic_id:
      wn_title_to_index[webnovel_title] = counter
      index_to_wn_title[counter] = webnovel_title
      counter+=1
    return tokenized_descriptions    

def build_inverted_index(msgs: List[dict]) -> dict:
    """Builds an inverted index from either webnovels or fanfics.

    Arguments
    =========
    msgs: List[Dict{id:int, description:str}]
        Each message in this list already has a 'tokenized_description'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
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
    #raise NotImplementedError()

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

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
    # TODO-5.1
    idf = {}
    for term in inv_idx: 
      term_df = len(inv_idx[term])

      if term_df/n_docs <= max_df_ratio and term_df >= min_df:
        idf[term] = math.log2(n_docs/(1 + term_df))
    return idf

def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    # TODO-6.1
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
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    # TODO-7.1
    doc_scores = {}
    for query_word in query_word_counts: 
      query_word_count = query_word_counts[query_word]
      for (doc, tf) in index[query_word]:
        if doc not in doc_scores: 
          doc_scores[doc] = tf*idf[query_word]*query_word_count*idf[query_word]
        else:
          doc_scores[doc] += tf*idf[query_word]*query_word_count*idf[query_word]
    return doc_scores

# want to build a cosine similarity matrix of webnovels_descriptions x fanfics_descriptions
def index_search(
    webnovel_toks: Dict[str, str],
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    webnovel_toks: string,
        The webnovel.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """

    # TODO-8.1
    webnovel_unique_toks = list(set(webnovel_toks))
    webnovel_word_counts = dict(Counter(webnovel_unique_toks)) 

    norm = 0
    for term in webnovel_unique_toks: 
      if term in idf and term in webnovel_word_counts:
        norm += (webnovel_word_counts[term]*idf[term])**2
      elif term not in idf and term in webnovel_word_counts:
        del webnovel_word_counts[term]
    norms = math.sqrt(norm) * doc_norms

    doc_scores = score_func(webnovel_word_counts, index, idf)

    cossim = []
    for doc in doc_scores:
      cossim.append((doc_scores[doc]/norms[doc], doc))

    result = sorted(cossim, key=lambda x: x[0], reverse=True)
    return result
    

def build_sims_cos(webnovels_tokenized, fanfic_inv_index, fanfic_idf, fanfic_doc_norms, score_func, input_get_sim_method):
    """Returns a movie_sims matrix of size (num_movies,num_movies) where for (i,j):
        [i,j] should be the cosine similarity between the movie with index i and the movie with index j
        
    Note: You should set values on the diagonal to 1
    to indicate that all movies are trivially perfectly similar to themselves.
    
    Params: {n_mov: Integer, the number of movies
             movie_index_to_name: Dictionary, a dictionary that maps movie index to name
             input_doc_mat: Numpy Array, a numpy array that represents the document-term matrix
             movie_name_to_index: Dictionary, a dictionary that maps movie names to index
             input_get_sim_method: Function, a function to compute cosine similarity}
    Returns: Numpy Array 
    """
    # TODO-5.4
    webnovel_sims = {} # key - webnovel id, value = list of sorted fanfics by similarity

    for webnovel in webnovels_tokenized:
        cossims = input_get_sim_method(webnovel['tokenized_description'])
        webnovel_sims[webnovel['id']] = cossims

    return webnovel_sims
    
    # for i in range(n_mov):
    #   for j in range(i, n_mov):
    #     if i==j:
    #       movie_sims[i, j] = 1
    #     else:
    #       movie_sims[i, j] = input_get_sim_method(movie_index_to_name[i], movie_index_to_name[j], input_doc_mat, movie_name_to_index)
    #       movie_sims[j, i] = movie_sims[i, j]

    # return movie_sims
    

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
    msgs: List[List],
    ins_cost_func: int,
    del_cost_func: int,
    sub_cost_func: int,
) -> List[Tuple[int, str]]:
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
    # TODO-1.2
    lst_of_tups = []
    
    #print(msgs[0])      # List of titles for a single webnovel
    for webnov_title_lst in msgs:
      wn_title = webnov_title_lst[0]
      #novel_title_ind = wn_title_to_index[webnov_title_lst[0]]
     #print("HERES THE INDEX", novel_title_ind)
      score = edit_distance(query, wn_title, ins_cost_func, del_cost_func, sub_cost_func)
      lst_of_tups.append((score, wn_title))

    sort_lst = sorted(lst_of_tups, key = lambda x: x[0])[:10]

    # lst_of_titles = [t[1] for t in sort_lst]
    
    # lst_dict_titles = []
    # for title in lst_of_titles:

    return [t[1] for t in sort_lst]

  
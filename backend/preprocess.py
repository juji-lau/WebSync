"""
Name: preprocess.py
Author: Juji Lau

Description: 
This file stores the functions for parsing and preprocessing the fanfiction and 
webnovel data from the JSON.
"""


from collections.abc import Callable
import numpy as np
import math
from collections import defaultdict, Counter
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


# Set up logging for the file
logging = logger.get_logger()

""" ------------------------------ GETTING DATA ------------------------------------ """
"""
Constants that store data: 
    fic_id_to_index (dict[fanfiction_id:int, index:int]): maps the fanfiction id to a zero-based index. 
    webnovel_title_to_index (dict): 
    index_to_fic_id (dict[index:int, fanfiction_id:int]): maps a zero-based index to the fanfiction id.
"""
fic_id_to_index = {}
index_to_fic_id = {}
webnovel_title_to_index = {}
index_to_webnovel_title = {}
fanfic_id_to_popularity = {}

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
    fanfics = []
    # files is a list of dictionaries.  list[dict(fanfic_id, description)]
    files = ['fanfic_G_2019_processed-pg1.json', 'fanfic_G_2019_processed-pg2.json', 'fanfic_G_2019_processed-pg3.json']
    for file in files:
        filepath = f"../dataset/{file}"
        with open(filepath, 'r') as f:
            fanfics = fanfics + json.load(f)
    return fanfics

def get_webnovel_data():
    """"
    Gets the webnovel data in the form: 

    [
        {
            "titles": ["Title 1", "Title 2", ...], (list[str])
            "original_lang": "English",           (str) 
            "rank": 153,                          (int)
            "description": "....",                (str)
            "authors": ["author 1"],              (list[str])
            "genres": ["Horror", ...],            (list[str])
            "tags": ["Battle Competition", ...],  (list[str])
        }, ...
    ]
    """
    webnovels = []
    # files is a list of dictionaries.  list[dict(fanfic_id, description)]
    files = ['novel_info.json']
    for file in files:
        filepath = f"../dataset/{file}"
        with open(filepath, 'r') as f:
            webnovels = webnovels + json.load(f)
    return webnovels

""" ------------------------------ PREPROCESSING ------------------------------------ """
def tokenize(text: str) -> list[str]:
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    list[str]
        A list of strings representing the words in the text.
    """
    tokenized_with_stop_words = re.findall("[A-Za-z]+", text.lower())
    tokenized = []
    for token in tokenized_with_stop_words:
        if token in stop_words: 
            pass
        else:
            tokenized.append(token)
    return tokenized

# files is a list of dictionaries.  list[dict(fanfic_id, description)]
def tokenize_fanfics(tokenize_method: Callable[[str], list[str]], 
    input_fanfics: list[dict[str, str]], ) -> list[str]:
    """Returns a list of tokens contained in an entire list of fanfics. 
       Also builds fic_id_to_index{}.  

    Parameters
    ----------
    tokenize_method : Callable[[str], list[str]]
        A method to tokenize a string into a list of strings representing words.
    input_fanfics : list[dict[]].  See get_fanfic_data() for more info
        A list of fanfiction dictionaries 
        (specified as a dictionary with ``id``, ``description``, and other tags specified above).
    
    Returns
    -------
    list[dict{fanfic_id:int, tokenized_description:list[str]}]
        A list of dictionaries, where each dictionary is a single fanfiction, with an id and a tokenized description.
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
        
        # popularity = fanfic_dict['hits'] + fanfic_dict['kudos'] + fanfic_dict['comments'] + fanfic_dict['bookmarks']
        if fanfic_dict['hits'] > 0:
            popularity = (fanfic_dict['kudos'] * fanfic_dict['chapters'])/fanfic_dict['hits']
        else: 
            popularity = 0
        fanfic_id_to_popularity[counter] = popularity
        
        counter+=1
    return tokenized_descriptions  

def tokenize_webnovels(tokenize_method: Callable[[str], list[str]], 
    input_webnovels: list[dict[str, str]], ) -> list[str]:
    """Returns a list of tokens contained in an entire list of webnovels. 
      Creates an id mapping for webnovels, then populates: webnovel_title_to_index{} 
      and index_to_webnovel_title{}  

    Parameters
    ----------
    tokenize_method : Callable[[str], list[str]]
        A method to tokenize a string into a list of strings representing words.
    input_webnovels : list[dict{}]  See get_webnovel_data() for more info.
        A list of webnovel dictionaries 
        (specified as a dictionary with ``title``, ``description``, and other tags specified above).

    Returns
    -------
    list[dict{webnovel_index : int, tokenized_descriptions:str}]
        A list of dictionaries, where key == webnovel_index, and value == tokenized description
        Note: the webnovel_index is an index, which can be mapped to the *first* title of the associated webnovel.
    """
    counter = 0
    tokenized_descriptions = []
    for webnovel_dict in input_webnovels:
        webnovel_title = webnovel_dict['titles'][0]
        webnovel_description = webnovel_dict['description']
        tokenized_descriptions.append({"index":counter, "tokenized_description":tokenize_method(webnovel_description)})
        webnovel_title_to_index[webnovel_title] = counter
        index_to_webnovel_title[counter] = webnovel_title
        counter+=1
    return tokenized_descriptions
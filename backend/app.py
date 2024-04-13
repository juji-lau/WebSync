import json
import os
from flask import Flask, render_template, request, session
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from scratch import edit_distance_search
from scratch import insertion_cost, deletion_cost, substitution_cost
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
novel_json_file_path = os.path.join(current_directory, 'novel_info.json')

cossim_json_file_path = os.path.join(current_directory, 'webnovel_to_fanfic_cossim.json')

"""========================== Gathering data: ============================="""
def getKeyInfo(data,key):
    lst = []
    for i in range(len(data)):
        lst.append(data[i][key])
    return lst

def getTitleInfo(data):
    lst = []
    for i in range(len(data)):
        lst.append(data[i]['titles'][0])
    return lst

with open(novel_json_file_path, 'r') as file:
    novel_data = np.array(json.load(file))
    novel_titles = getKeyInfo(novel_data,'titles')
    novel_descriptions = getKeyInfo(novel_data,'description')
    novel_title_to_index = {}
    for i in range (len(novel_titles)):
        novel_title_to_index[novel_titles[i][0]] = i
        
fanfics = {}
fanfic_files = ['fanfic_G_2019_processed-pg1.json', 'fanfic_G_2019_processed-pg2.json', 'fanfic_G_2019_processed-pg3.json']
for file in fanfic_files:
    file = os.path.join(current_directory, file)
    with open(file, 'r') as f: 
        temp_fanfic_list = json.load(f)

        for fanfic_info in temp_fanfic_list:
            fanfics[fanfic_info['id']] = fanfic_info

with open(cossim_json_file_path, 'r') as file: 
    file_contents = json.load(file)
    cossims = file_contents['cossims']
    webnovel_title_to_index = file_contents['webnovel_title_to_index']
    index_to_fanfic_id = file_contents['index_to_fanfic_id']

app = Flask(__name__)

app.secret_key = 'BAD_SECRET_KEY'
CORS(app)

""" ========================= Backend stuff: ============================"""
def json_search(query):
    """ Searches the webnovel database for a matching webnovel to the user typed query 
    using string matching.  
    Called for every character typed in the search bar.

    Argument(s):
    query:str - what the user types when searching for a webnovel

    Return(s):
    matches: Dict{ str: str} - a list of matching webnovel dictionaries to the query. 
        Each dictionary includes the webovel title and description currently.  
    """
    print("a1. In json_search(query) in app.py          No app.route()")
    matches = []
    for i in range (len(novel_titles)):
        if query.lower() in novel_titles[i][0].lower() and query != "":
            matches.append({'title': novel_titles[i][0],'descr':novel_descriptions[i]})
    return matches

def user_description_search(user_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    docs_tfidf = vectorizer.fit_transform(novel_descriptions)

    svd = TruncatedSVD(n_components=50)
    docs_svd = svd.fit_transform(docs_tfidf)
    user_tfidf = vectorizer.transform([user_description])
    user_svd = svd.transform(user_tfidf)
    
    sims = cosine_similarity(user_svd, docs_svd).flatten()
    result_indices = np.argsort(sims)
    matches = []
    for i in range(1,6):
        result_index = result_indices[-i]
        matches.append({'title': novel_titles[result_index][0], 
                        'descr': novel_descriptions[result_index]})
    return matches


@app.route("/fanfic-recs/")
def recommendations(): 
    """
    Called when the user clicks "Show Reccommendations"
    Links to showResults(title) in base.html
    """
    print("a2. In recomendations() app.py           app.route(/fanfic-recs/)")
    return webnovel_to_top10fics(session['title'])

def webnovel_to_top10fics(webnovel_title):
    """
    Called when the user clicks "Show Recommendations"
    input: webnovel_title --> the title of the user queried webnovel
    output: the top 10 fanfiction information. Can include: 
        - fanfic_id
        - fanfic_titles
        - descriptions
        - etc.
    """
    print("a3. In webnovel_to_top10fanfictions() app.py         No app.route()")
    webnovel_index = webnovel_title_to_index[webnovel_title]
    sorted_fanfics_tuplst = cossims[str(webnovel_index)]
    top_10 = sorted_fanfics_tuplst[:10]
    top_10_fanfic_indexes = [t[1] for t in top_10]
    top_10_fanfics = []
    for i in top_10_fanfic_indexes:
        fanfic_id = index_to_fanfic_id[str(i)]
        info_dict = {}
        info_dict["fanfic_id"] = fanfic_id                              # get fanfic id
        info_dict["description"] = fanfics[fanfic_id]['description']    # get description
        info_dict["title"] = fanfics[fanfic_id]["title"]                # get title
        top_10_fanfics.append(info_dict)
    return top_10_fanfics
    

@app.route("/")
def home():
    print("a4. In home() in app.py          app.route(/)")
    print(novel_titles[0])
    session['title'] = novel_titles[0][0]
    session['title-index'] = 0
    session['tags'] = None
    return render_template('home.html',title="sample html")

selectedNovel = ""
@app.route("/results/")
def results():
    """ Called when the user clicks the --> arrow on the home page."""
    print("a5. In results() in app.py           app.route(/results)")
    session['tags'] = None
    return render_template('base.html',title="sample html")

@app.route("/titleSearch")
def titleSearch():
    """
    Gets the user typed query, and calls json_search to return relevant webnovels.
    Links to function filterText(id) in home.html.
    """
    print("a6. In titleSearch() in app.py.          app.route(/titleSearch)")
    text = request.args.get("inputText")
    return json_search(text)

@app.route("/descrSearch")
def descrSearch():
    """
    Gets the user typed query, and calls json_search to return relevant webnovels.
    Links to function filterText(id) in home.html.
    """
    print("a6. In descrSearch() in app.py.          app.route(/descrSearch)")
    text = request.args.get("inputText")
    return user_description_search(text)

@app.route("/setNovel")
def setNovel():
    """ 
    Returns the user selected wbenovel.
    Links to function titleButtonEventListener(e) in home.html 
    """
    print("a7. In setNovel() in app.py          app.route(/setNovel)")
    selectedNovel = request.args.get("title")
    session['title'] = request.args.get("title")
    session['title-index'] = novel_title_to_index[selectedNovel]
    returnDict = {'title': selectedNovel}
    return returnDict

@app.route("/getNovel")
def getNovel():
    """
    Retrieves the selected webnovel title, author, description, and genres from the second page.
    Links to function setup() in base.html
    Called as soon as user enters the second page

    Returns: 
    returnDict: Dict{
        title: webnovel title
        descr: webnovel description
        author: The first listed author of the webnovel
        genres: All the genres of the webnovel
    }
    """
    print("a8. In getNovel() in app.py          app.route(/getNovel)")
    index = session['title-index']
    returnDict = {'title': novel_titles[index],
                  'descr':novel_descriptions[index],
                  'author':novel_data[index]['authors'][0],
                  'genres': novel_data[index]['genres']}
    return returnDict

@app.route("/addTag")
# whenever the user adds a tag, this is called
def addTag():
    """ Links to function addTag(e) in home.html"""
    print("a9. in addTag() in app.py            app.route(/addTag")
    newTag = request.args.get("tag")
    # If a user adds more than one tag, including empty tags
    if session.get('tags') != None:
        print("OooooOoooooooo")
        session['tags'].append(newTag)
    # when the user adds the first tag, including empty tags
    else:
        session['tags'] = []
        print("BYEEEEEEEEEE")
        session['tags'].append(newTag)
    session.modified = True
    return {'tags': newTag}

@app.route("/removeTag")
def removeTags():
    """ Links to function addTag(e) in home.html """
    print("a10. in removeTags() in app.py().            app.route(/removeTag) ")
    tag = request.args.get("tag")
    print("Tag removed: ", tag)
    print("Current tags: ", session['tags'])
    session['tags'].remove(tag)
    print(session['tags'])
    session.modified = True
    return {'tags': tag}

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
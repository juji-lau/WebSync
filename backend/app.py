import json
import os
from flask import Flask, render_template, request, session
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
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


def json_search(query):
    matches = []
    for i in range (len(novel_titles)):
        if query.lower() in novel_titles[i][0].lower() and query != "":
            matches.append({'title': novel_titles[i][0],'descr':novel_descriptions[i]})
    return matches


@app.route("/fanfic-recs")
def webnovel_to_top10fics(webnovel_title):
    """
    input: webnovel_title --> the title of the user queried webnovel
    output: the top 10 fanfiction information. Can include: 
        - fanfic_id
        - fanfic_titles
        - descriptions
        - etc.
    """
    webnovel_index = webnovel_title_to_index[webnovel_title]
    sorted_fanfics_tuplst = cossims[webnovel_index]
    top_10 = sorted_fanfics_tuplst[:10]
    top_10_fanfic_indexes = [top_10[1] for t in top_10]
    top_10_fanfics = []
    for i in top_10_fanfic_indexes:
        fanfic_id = index_to_fanfic_id[i]
        info_dict = {}
        info_dict["fanfic_id"] = fanfic_id                              # get fanfic id
        info_dict["description"] = fanfics[fanfic_id]['description']    # get description
        info_dict["title"] = fanfics[fanfic_id]["title"]                # get title
        top_10_fanfics.append(info_dict)
    return top_10_fanfics
    

@app.route("/")
def home():
    session['title-index'] = 0
    session['tags'] = None
    return render_template('home.html',title="sample html")

selectedNovel = ""
@app.route("/results/")
def results():
    session['tags'] = None
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)
    #Return a List (dictionary: { title, description })
    # list_dicts = []

    # novel_dicts = []
    # for novel in novel_titles:
    #     novel_dicts.append({'text': novel})
    # tup_list = edit_distance_search(text, novel_dicts, insertion_cost, deletion_cost, substitution_cost)
    # print(tup_list[0])
    # for tup in tup_list:
    #     print(tup)
    #     score = tup[0]
    #     title = tup[1]
    #     # Create a dictionary for each tuple
    #     dict_title = {'title': title, 'score': score}
    #     # Append the dictionary to the list
    #     list_dicts.append(dict_title)
    # return list_dicts


@app.route("/setNovel")
def setNovel():
    selectedNovel = request.args.get("title")
    # session['title'] = request.args.get("title")
    session['title-index'] = novel_title_to_index[selectedNovel]
    returnDict = {'title': selectedNovel}
    return returnDict

@app.route("/getNovel")
def getNovel():
    index = session['title-index']
    returnDict = {'title': novel_titles[index],
                  'descr':novel_descriptions[index],
                  'author':novel_data[index]['authors'][0],
                  'genres': novel_data[index]['genres']}
    return returnDict

@app.route("/addTag")
def addTag():
    print("add")
    newTag = request.args.get("tag")
    if session.get('tags') != None:
        print("hi")
        session['tags'].append(newTag)
    else:
        session['tags'] = []
        session['tags'].append(newTag)
    session.modified = True
    return {'tags': newTag}

@app.route("/removeTag")
def removeTags():
    print("remove")
    tag = request.args.get("tag")
    print(tag)
    print(session['tags'])
    session['tags'].remove(tag)
    print(session['tags'])
    session.modified = True
    return {'tags': tag}

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
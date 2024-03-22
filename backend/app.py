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



def getKeyInfo(data,key):
    lst = []
    for i in range(len(data)):
        lst.append(data[i][key])
    return lst

# Assuming your JSON data is stored in a file named 'init.json'
with open(novel_json_file_path, 'r') as file:
    novel_data = np.array(json.load(file))
    novel_titles = getKeyInfo(novel_data,'titles')
    novel_descriptions = getKeyInfo(novel_data,'description')
    novel_title_to_index = {}
    for i in range (len(novel_titles)):
        novel_title_to_index[novel_titles[i][0]] = i

app = Flask(__name__)

app.secret_key = 'BAD_SECRET_KEY'
CORS(app)


def json_search(query):
    matches = []
    for i in range (len(novel_titles)):
        if query.lower() in novel_titles[i][0].lower() and query != "":
            matches.append({'title': novel_titles[i],'descr':novel_descriptions[i]})
    return matches

@app.route("/")
def home():
    session['title-index'] = 0
    session['tags'] = None
    return render_template('home.html',title="sample html")

@app.route("/results/")
def results():
    session['title-index'] = 0
    session['tags'] = None
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")

    return json_search(text)

    # Return a List (dictionary: { title, description })
    # list_dicts = []
    # list_titles = edit_distance_search(text, novel_titles, insertion_cost, deletion_cost, substitution_cost )
    # for title in list_titles:
    #     # Create a dictionary for each tuple
    #     dict_title = {'title': title}
    #     # Append the dictionary to the list
    #     list_dicts.append(dict_title)
    # return list_dicts


@app.route("/setNovel")
def setNovel():
    selectedNovel = request.args.get("title")
    session['title-index'] = novel_title_to_index[selectedNovel]
    returnDict = {'title': selectedNovel}
    return returnDict

@app.route("/getNovel")
def getNovel():
    returnDict = {'title': novel_titles[session['title-index']]}
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
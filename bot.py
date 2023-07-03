import os
from flask import Flask, render_template, request
# import aiml
#from flask_session import Session
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
#import spacy
import requests
import string
import urllib.parse
from datetime import date
from gingerit.gingerit import GingerIt
# from nltk.corpus import wordnet
# from neo4j import GraphDatabase
#from neo4j import GraphDatabase
#from nltk.sentiment import SentimentIntensityAnalyzer
from py2neo import Graph,NodeMatcher
# from pyaiml21 import Kernel
from aiml import Kernel
from glob import glob
from bs4 import BeautifulSoup
import pytholog as pl
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
#nltk.download('vader_lexicon')
#nltk.download('')
from nltk import pos_tag
from gingerit.gingerit import GingerIt
#from nlp_portion import NER
#from nlp_portion imppiort GR
#from nlp_portion import is_question
graph = Graph("bolt://localhost:7687",auth=("neo4j","12345678"))
#river = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

app = Flask(__name__)

sia = SentimentIntensityAnalyzer()
Bot = Kernel()
def learn_aiml():
   for file in glob("./data/*.aiml"):
        print("learning",file)
        Bot.learn(file)
learn_aiml()
userID="md1818"
Bot.setPredicate(name="username",value="Ali")
Bot.setPredicate("name","mdBOt")

@app.route('/')
def login():
    return render_template('login.html')

@app.route("/signup", methods=['POST'])
def getvalue():
    username = request.form.get('username')
    email = request.form.get('email')
    pass1 = request.form.get('pass1')
    graph.run(f"CREATE (n:person{{name: \"{username}\", email: \"{email}\", password: {pass1}}})")
    return render_template("login.html")

@app.route("/login", methods=['POST'])
def login_user():
    email = request.form.get('email')
    pass1 = request.form.get('password')
    print(email,pass1)
    email_ver0 = graph.run(f"MATCH (n:person{{email: \"{email}\", password: {pass1}}}) return n")
    emailver = list(email_ver0)
    print(emailver)
    if emailver:
        return render_template('home.html')
    else:
        return render_template('login.html')

@app.route('/registration')
def about():
    return render_template('registration.html')
@app.route("/home")
def home():
    return render_template("home.html")



model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_weights('gender_model_weights.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_gender(name):
    encoding = tokenizer([name], truncation=True, padding=True)
    input_dataset = tf.data.Dataset.from_tensor_slices(dict(encoding)).batch(1)
    predictions = model.predict(input_dataset)
    predicted_label = tf.argmax(predictions.logits, axis=1)[0].numpy()
    gender = "male" if predicted_label == 0 else "female"
    return gender


def get_gender_prediction(text):
    gender = predict_gender(text)
    if gender == "male":
        return f"Yes, he is male."
    else:
        return f"Yes, she is female."



def get_part_of_speech(word):
    # Tokenize the word
    tokens = nltk.word_tokenize(word)
    # Perform part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)
    # Check if the word is a noun, verb, or adjective
    for token, pos in pos_tags:
        if token.lower() == word.lower():
            if pos.startswith("NN"):
                return "is noun"
            elif pos.startswith("VB"):
                return " is verb"
            elif pos.startswith("JJ"):
                return "is adjective"

    # Return the default response
    return "not noun, verb, or adjective"


def getWordnet(word):
    wn_definition = ""
    synsets = wn.synsets(word)
    if synsets:
        synset = synsets[0]
        wn_definition = synset.definition()
    return wn_definition


def search_wikipedia(query):
    words = query.split()
    last_two_words = words[-2:]  # Extract last two words
    search_terms = [string.capwords(word) for word in last_two_words]
    encoded_terms = [urllib.parse.quote(term) for term in search_terms]

    line_count = 0

    for term in encoded_terms:
        url = "https://en.wikipedia.org/wiki/" + term
        url_open = requests.get(url)
        soup = BeautifulSoup(url_open.content, 'html.parser')
        paragraphs = soup.find_all('p')
        response=""
        if paragraphs:
            for paragraph in paragraphs:
                lines = paragraph.get_text().split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        response = line
                        return response

    return "No information found."
# Example usage:
def correct_spelling(text):
    ginger_parser = GingerIt()
    corrected_text = ginger_parser.parse(text)['result']
    return corrected_text



@app.route("/get")
def get_bot_response():
    #response = ""
    query = request.args.get('msg')
    print("query",query)
    response = Bot.respond(query)
    #query1 = correct_spelling(query)
    #print("query",query)
    current_date = date.today().isoformat()
    result = graph.run(
        """
        MERGE (c:Chat {date: $dateParam})
        ON CREATE SET c.queries = [], c.responses = []
        RETURN c
        """,
        dateParam=current_date
    )

    chat_node = result.evaluate()

    if chat_node is None:
        # Node creation failed, handle the error
        # For example, return an error response or log the issue
        return "Failed to create or retrieve chat node"
    chat_node['queries'].append(query)
    #chat_node['responses'].append(response)
    graph.push(chat_node)

    sentiment_score = sia.polarity_scores(query)
    compound_score = sentiment_score['compound']
    #response = Bot.respond(query)
    if compound_score >= 0.05:
        sentiment_label = 'positive'
    elif compound_score <= -0.05:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'

    if sentiment_label == 'positive':
        response = "its great brooo"
    elif sentiment_label == 'negative':
        response = "I'm sorry to hear that. How can I assist you further?"  # Negative sentiment response
    else:
        b = Bot.getPredicate("gender")
        print("mygendername:",b)
        predict= None
        if b!="":
            predict = get_gender_prediction(b)
            print("fromgender:",predict)
            Bot.setPredicate("predict",predict)
            response=Bot.respond(query)

        a=Bot.getPredicate("wordspeech")
        print("a :",a)
        speech = None
        if a !="":
            speech=get_part_of_speech(a)
            print("part:",speech)
            Bot.setPredicate("speech",speech)
            response=Bot.respond(query)

        # c=Bot.getPredicate("lovey")
        # print("fromwordnet",c)
        # wordnet=None
        # if c !="":
        #     wordnet = getWordnet(c)
        #     print("wordnet",wordnet)
        #     Bot.setPredicate("wordnet", wordnet)
        #     response = Bot.respond(query)

        word = Bot.getPredicate("searchWord")
        print("word",word)
        definition = None
        if word != "":

            definition = getWordnet(word)
            print("defnition frm worned : ", definition)
            if not definition:
                definition = search_wikipedia(word)
                print("fromwikipedia", definition)
            Bot.setPredicate("definition", definition)
            response = Bot.respond(query)

        new_kb = pl.KnowledgeBase("social network")
        knowledgeBase = ["male(ali)", "male(ahmed)", "father(X,Y):-parent(X,Y),male(X)",
                         "female(jia)", "male(ahmed)", "mother(X,Y):-parent(X,Y),female(X)" ]

        p1 = Bot.getPredicate("person1")
        print("p1 :", p1)
        p2 = Bot.getPredicate("person2")
        rel = Bot.getPredicate("relation")
        facts = rel+"("+p1+","+p2+")"

        knowledgeBase.insert(0, facts)
        new_kb(knowledgeBase)
        y = new_kb.query(pl.Expr("father(X, Y)"))[0]

        Bot.setPredicate("father", str(y))
        Bot.setPredicate("prologFact", facts)
        response = Bot.respond(query)

        if response:
            print("bot>",response)


            chat_node['responses'].append(response)
            graph.push(chat_node)


    return response




if __name__ == "_main_":
    # app.run()
    app.run(host='0.0.0.0', port='5000')
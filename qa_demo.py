import csv
import sqlite3
from flask import Flask, g
from flask import request
from flask import render_template

from qa_model.qa_api import myQAModel

qaModel = myQAModel()

app = Flask(__name__)
SQLITE_DB_PATH = 'test2.db'
SQLITE_DB_SCHEMA = 'create_db.sql'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(SQLITE_DB_PATH)
        # Enable foreign key check
        db.execute("PRAGMA foreign_keys = ON")
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST', 'GET'])
def ask():
    error = None

    if request.method == 'POST':
        context = request.form['context']
        question = request.form['question']

    contexts = [context]
    questions = [question]
    Y_str, Y_end = qaModel.infer(contexts, questions, getIndex=True)
    answers = [contexts[i][Y_str[i]:Y_end[i]] for i in range(len(contexts))]

    # Get the database connection
    db = get_db()

    # Update query history
    # with db:
    #     db.execute('INSERT INTO query_histories(context, question, answer_start, answer_end) VALUES (?,?,?,?)',
    #     (context, question, ))

    return 'Answer is: %s' % answers[0]


if __name__ == '__main__':

    app.run(debug=True)

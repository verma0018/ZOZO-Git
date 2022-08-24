import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request

app = Flask(__name__,template_folder='templates')
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

questions = {
    1: 'Do you have interest in Human anatomy?',
    2: 'Do you have interest in Human Psychology?',
    3: 'Do you have interest in Designing of Drugs?',
    4: 'Do you have interest in Human Physiology?',
    5: 'Do you have interest in Machinery and manufacturing of goods',
    6: 'Do you have an interest in making all kind of devices and equipment?',
    7: 'Do you have an interest in developing software applications?',
    8: 'Do you have an interest in making of buildings, roads and bridges?',
    9: 'Do you have an interest in designing and developing chemicals ?',
    10: 'Do you have an interest in Space and Science?',
    11: 'Do you have an interest in providing people with knowledge and teaching them in something you are good at?',
    12: 'Do you have an interest in being a part of defense?',
    13: 'Do you have an interest in Literature?',
    14: 'Do you have an interest in Economy of the world?',
    15: 'Do you have an interest in working with the accounts of a company or an individual?',
    16: 'Do you have an interest in Practicing Law?',
    17: 'Do you have an interest in Managing projects and operations?',
    18: 'Do you have an interest in selling product and services by researching and advertising?',
    19: 'Do you have an interest in making income on your own?',
    20: 'Do you have an interest in working for the public?',
    21: 'Do you have an interest in Housing management?',
    22: 'Do you have an interest in Journalism?',
    23: 'Do you have an interest in the creation a curation of content, information (digital or physical)?',
    }

characters = [
    {'name': 'MBBS',                            'answers': {1: 1, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Therapist',                       'answers': {1: 0, 2: 1, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Pharmacy',                        'answers': {1: 0, 2: 0, 3: 1, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Physio-therapist',                'answers': {1: 0, 2: 0, 3: 0, 4: 1, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Mechanical Engineering',          'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:1, 6:0.25, 7:0, 8:0, 9:0, 10:0.5, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0.5, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Electronic Engineering',          'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:1, 7:0, 8:0, 9:0, 10:0.5, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0.5, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Computer Science Engineering',    'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:1, 8:0, 9:0, 10:0.5, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0.5, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Civil Engineering',               'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:1, 9:0, 10:0.5, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0.5, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Chemical Engineering',            'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:1, 10:0.25, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0.25, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Aeronautics Engineering',         'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0.25, 6:0.5, 7:0.5, 8:0, 9:0, 10:1, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0.25, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Teacher',                         'answers': {1: 0, 2: 0.25, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0.5, 11:1, 12:0, 13:0.75, 14:0.75, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0.5, 21:0, 22:0, 23:0.25}},
    {'name': 'Defense',                         'answers': {1: 0.25, 2: 0.25, 3: 0.25, 4:0.25,5:0.25, 6:0.25, 7:0.25, 8:0.25, 9:0.25, 10:0.25, 11:0, 12:1, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0.25, 20:0.25, 21:0, 22:0, 23:0}},
    {'name': 'Author',                          'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0.25, 11:0.5, 12:0, 13:1, 14:0, 15:0, 16:0, 17:0.25, 18:0, 19:0, 20:0, 21:0, 22:0.25, 23:0.5}},
    {'name': 'Economist',                       'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:1, 15:0.25, 16:0, 17:0.25, 18:0, 19:1, 20:0, 21:0, 22:0.5, 23:0}},
    {'name': 'Accounting',                      'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0.25, 14:0.75, 15:1, 16:0.5, 17:0.25, 18:0.5, 19:0.25, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Attorney',                        'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0.75, 14:0, 15:0.5, 16:1, 17:0, 18:0, 19:0.25, 20:0.75, 21:0, 22:0.5, 23:0}},
    {'name': 'Management',                      'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0.5, 12:0, 13:0.5, 14:0, 15:0, 16:0.5, 17:1, 18:0.25, 19:0.75, 20:0.5, 21:0.75, 22:0.25, 23:0}},
    {'name': 'Marketing',                       'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0.25, 14:0.5, 15:0.75, 16:0, 17:0.75, 18:1, 19:0.75, 20:0, 21:0, 22:0, 23:0.75}},
    {'name': 'Civil Services',                  'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0.25, 11:0, 12:0.75, 13:0.75, 14:0.75, 15:0.25, 16:0.75, 17:0.5, 18:0, 19:0.5, 20:1, 21:0, 22:0.5, 23:0}},
    {'name': 'Hotel Management',                'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0.25, 14:0.25, 15:0.25, 16:0, 17:0.5, 18:0.25, 19:0.25, 20:0.25, 21:1, 22:0, 23:0.5}},
    {'name': 'Journalist',                      'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0.25, 11:0, 12:0, 13:0.75, 14:0.75, 15:0, 16:0.25, 17:0.75, 18:0.5, 19:0.5, 20:0.75, 21:0, 22:1, 23:0.5}},
    {'name': 'Content Creator',                 'answers': {1: 0, 2: 0.25, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0.75, 14:0, 15:0, 16:0, 17:0.25, 18:0.75, 19:0.75, 20:0.25, 21:0, 22:0, 23:1}},
    {'name': 'Engineer',                        'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0.75, 6:0.75, 7:0.75, 8:0.75, 9:0.75, 10:0.75, 11:0.5, 12:0.5, 13:0.5, 14:0, 15:0, 16:0, 17:0.5, 18:0, 19:0.75, 20:0.5, 21:0, 22:0.5, 23:0.5}},
    {'name': 'Teacher',                         'answers': {1: 0, 2: 0.5, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0.75, 11:1, 12:0, 13:1, 14:0.5, 15:0.5, 16:0, 17:0.75, 18:0, 19:0.75, 20:0.75, 21:0, 22:0, 23:0.5}},
    {'name': 'Doctor',                          'answers': {1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0.5, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.5, 18:0, 19:0, 20:0, 21:0.5, 22:0, 23:0}},
    {'name': 'Entrepreneur',                    'answers': {1: 0, 2: 0.5, 3:0, 4:0, 5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0.25, 12:0, 13:0.75, 14:0.5, 15:0.5, 16:0, 17:0.75, 18:1, 19:1, 20:0.5, 21:0, 22:0, 23:0.75}},
]

questions_so_far = []
answers_so_far = []



@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    global questions_so_far, answers_so_far
    hi = ['Hi', "hi"]
    start = ['start', 'Start']
    # question = questions()
    userText = request.args.get('msg')
    question = request.args.get('qnum')
    answer = request.args.get('answer') # get value from userText 
    val = request.args
    print("question from args",question)
    print("usertext",userText)
    print("args",request.args)
    print("type f val",type(val))
    # ans = eval(val)
    print("json",val.items())
    # print("json",ans)
    yesList = ['yes', 'YES', 'Y', 'y']
    noList = ['no','NO','Nope', 'n']
    maybeList = ['maybe','m','probably', 'M']
    maybenotList = ['maybe not','probably not','MN', 'mn']
    dontknowList = ['dont know','Dont know','dk', 'DK','Dk']
    if(userText in yesList):
        answer = 1
    elif(userText in noList):
        answer = 0.01
    elif(userText in maybeList):
        answer = 0.75
    elif(userText in maybenotList):
        answer = 0.25
    elif(userText in dontknowList):
        answer = 0.5
    else:
        answer = 0
    if(answer == 0):
        if(userText not in hi and userText not in start and question =='' or question == 'undefined'):
            return ['error','Sorry I could not understand that, let`s start again...Hi I am zozo']
        if(userText not in hi and userText not in start):
            print("i am inside")
            v = int(question)
            return ['error','To start your career counseling session please enter start!* Do remember that the answers to the upcoming questions should be given in the form of yes(y), no(n), maybe(m), maybe not(mn), and don`t know(dk)', questions[v]]
        

        # if(userText not in )
        
    # elif(userText == 'done' or userText == 'stop'):
    #      break()
    # if(answer == 0):
    #     return "Sorry You entered something please enter something from the data given below."

    print("question",question)
    print("answer",answer)
    if question and answer:
        questions_so_far.append(int(question))
        answers_so_far.append(float(answer))
        print("questions so far",questions_so_far)
        print("questions so far",answers_so_far)
    
    
    
    probabilities = calculate_probabilites(questions_so_far, answers_so_far)
    print("probabilities", probabilities)

    questions_left = list(set(questions.keys()) - set(questions_so_far))
    print("questioons",questions_left)
    if len(questions_left) == 0:
        result = sorted(
            probabilities, key=lambda p: p['probability'], reverse=True)[0]
        result=result['name']
        return [result]
        # render_template('index.html', result=result['name'])
    else:
        #questio_text
        next_question = random.choice(questions_left)
        question_text=questions[next_question]
        print(next_question,question_text)
        return [question_text ,next_question]
        # render_template('index.html', question=next_question, question_text=questions[next_question])
    
    # return chatbot_response(userText)


def calculate_probabilites(questions_so_far, answers_so_far):
    probabilities = []
    for character in characters:
        probabilities.append({
            'name': character['name'],
            'probability': calculate_character_probability(character, questions_so_far, answers_so_far)
        })

    return probabilities


def calculate_character_probability(character, questions_so_far, answers_so_far):
    # Prior
    P_character = 1 / len(characters)

    # Likelihood
    P_answers_given_character = 1
    P_answers_given_not_character = 1
    for question, answer in zip(questions_so_far, answers_so_far):
        P_answers_given_character *= 1 - \
            abs(answer - character_answer(character, question))

        P_answer_not_character = np.mean([1 - abs(answer - character_answer(not_character, question))
                                          for not_character in characters
                                          if not_character['name'] != character['name']])
        P_answers_given_not_character *= P_answer_not_character

    # Evidence
    P_answers = P_character * P_answers_given_character + \
        (1 - P_character) * P_answers_given_not_character

    # Bayes Theorem
    P_character_given_answers = (
        P_answers_given_character * P_character) / P_answers

    return P_character_given_answers


def character_answer(character, question):
    if question in character['answers']:
        return character['answers'][question]
    return 0.5

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
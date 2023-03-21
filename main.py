#main.py
from flask import Flask, render_template, session, url_for, redirect, request
from inference import predict, BERTClassifier, BERTDataset
import time

app = Flask(__name__)
app.secret_key = 'this is super key'
app.config['SESSION_TYPE'] = 'filesystem'
userinfo = {'fu': 'fu'}
board = []

def print_emotion(label):
  if label == 0:
    label_name = '기쁨'
  if label == 1:
    label_name = '불안'
  if label == 2:
    label_name = '당황'
  if label == 3:
    label_name = '슬픔'
  if label == 4:
    label_name = '분노'
  if label == 5:
    label_name = '상처'
  return label_name

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('board.html')
    if request.method == 'POST':

        context1 = request.form['context1']
        context2 = request.form['context2']
        context3 = request.form['context3']

        con_list = []
        con_list.append(context1)
        con_list.append(context2)
        con_list.append(context3)

        answer = {0:'', 1:'', 2:''}
        for i, context in enumerate(con_list):
            pred = predict(context)
            answer[i] = print_emotion(pred)
        con1, con2, con3 = f'Q1: {con_list[0]}', f'Q2: {con_list[1]}', f'Q3: {con_list[2]}'
        answer1, answer2, answer3 = f'A1: 화자는 {answer[0]}을(를) 느끼고 있습니다.',f'A2: 화자는 {answer[1]}을(를) 느끼고 있습니다.',f'A3: 화자는 {answer[2]}을(를) 느끼고 있습니다.'
        return render_template('board.html', con1=con1, con2=con2, con3=con3, answer1=answer1, answer2=answer2, answer3=answer3)


if __name__ == '__main__':
    app.run(debug=True)

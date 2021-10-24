import os
import csv
import sys
import nltk
import random 
import itertools
import numpy as np
import pandas as pd
from math import log10
from sklearn import tree
from scipy import spatial
from numpy import loadtxt
from urllib import request
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.utils import resample
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#analyzer for count vectoriser
def stemmed_words(doc):
    return (sb_stemmer.stem(w) for w in analyzer(doc))

#calculate similarity between document and query
def csimilarity(document, query):
  v = CountVectorizer(stop_words=stopwords.words('english'), analyzer=stemmed_words)
  sentences = []
  sentences.append(query)
  sentences.append(document)
  try:
    fitted_v = v.fit_transform(sentences)
    similarity = cosine_similarity(fitted_v)
    similarity = similarity[1,0]
  except ValueError:
    similarity = 0
  return similarity

classification_data = []
analyzer = CountVectorizer().build_analyzer()
sb_stemmer = SnowballStemmer('english')

# ---- LOADING DATA ---- # 
data_folder = os.path.join(sys.path[0])
i2_dataset1 = pd.read_csv(os.path.join(data_folder, "qna_chitchat_friendly.tsv"), sep='\t')
i2_dataset2 = pd.read_csv(os.path.join(data_folder, "qna_chitchat_enthusiastic.tsv"), sep='\t')
i2_dataset3 = pd.read_csv(os.path.join(data_folder, "qna_chitchat_professional.tsv"), sep='\t')


i2_answers = [[] for x in range(3)]
i2_questions = i2_dataset1['Question'].values.tolist()
i2_answers[0] = i2_dataset1['Answer'].values.tolist()
i2_answers[1] = i2_dataset2['Answer'].values.tolist()
i2_answers[2] = i2_dataset3['Answer'].values.tolist()


for question in i2_questions:
  classification_data.append([question, "intent2"])


i3_data = []
with open(os.path.join(data_folder, "COMP3074-CW1-Dataset.csv"), encoding="utf-8") as csv_file:
  reader = csv.reader(csv_file)
  for row in reader:
    i3_data.append((row[1], row[2]))
    classification_data.append([row[1], "intent3"])

# ---- CLASSIFIER ---- # 
#oversampling intent 3 so that datasets are balanced
majority_class = [x for x in classification_data if x[1] == "intent2"]
minority_class = [x for x in classification_data if x[1] == "intent3"]
upsampled_class = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=123)
final_upsampled = majority_class + upsampled_class
new_data =  [x[0] for x in final_upsampled]
new_labels = [x[1] for x in final_upsampled]

#splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(new_data, new_labels, stratify=new_labels, test_size=0.25, random_state=42)
count_vect = CountVectorizer(stop_words=stopwords.words('english'))

#building classifier
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
X_train_tf = tfidf_transformer.transform(X_train_counts)
classifier = LogisticRegression(random_state=0).fit(X_train_tf, y_train)
#classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X_train_tf, y_train)

#evaluating on test data
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = classifier.predict(X_new_tfidf)

print(confusion_matrix(y_test, predicted))
print(accuracy_score(y_test, predicted))
print(f1_score(y_test, predicted, average=None))





user_name = ""
change_name_sentences =["call me", "change my name", "register my name as", "update my name", "make my name", "my name is"]
output_name_sentences = ["what is my name", "what's my name", "what my name is", "print out my name", "display my name", "me my name", "forgot my name", "can't remember my name", "give me my name"]


# ---- CHATBOT ---- # 
while(True):

  intent_1 = False

  #storing user's name 
  if user_name == "":
      print("BOT: Hey there! What's your name?")
      try:
        user_name = str(input())
      except KeyboardInterrupt:
        print("Bye!")
        sys.exit()
      print("BOT: Your name is " + user_name + ", let the bot know if you would like to change your name anytime. Start chatting, and type 'bye' when you want to quit.")

  #getting query from user
  try:
    query = str(input())
  except KeyboardInterrupt:
        print("BOT: Bye!")
        sys.exit()

  # --- INTENT 1 --- #
  #display name on demand
  for phrase in output_name_sentences:
    if phrase in query:
      print("BOT: Oh! Have you forgotten? Your name is " +  user_name + "!")
      intent_1 = True

  #change name 
  for phrase in change_name_sentences:
    if phrase in query:
      print("BOT: So, you'd like to change your name. Could you type out your new name please?")
      user_name = str(input())
      print("BOT: Hello " + user_name + "!")
      intent_1 = True
  
#intents 2 and 3

  if intent_1 == False:

    #classifying user query
    x = count_vect.transform([query])
    x_transformed = tfidf_transformer.transform(x)
    predicted = classifier.predict(x_transformed)
    
    # --- INTENT 2 --- #    
    if predicted == "intent2":
      q_similarities = []

      for document in i2_questions:
        similarity = csimilarity(document, query)
        q_similarities.append([similarity, document])

      q_similarities.sort(reverse=True)

      if q_similarities[0][0] >= 0.5:
        #picking random response related to similar question
        r = random.choice([i2_answers[0], i2_answers[1], i2_answers[2]])
        index = i2_questions.index(q_similarities[0][1])
        answer = r[index]
        print("BOT:", answer)
      else:
        print("BOT: I'm sorry, I am unable to answer this question at the moment.")
      
    # --- INTENT 3 --- #
    if predicted == "intent3":
      q_similarities = []

      for document in i3_data:
        similarity = csimilarity(document[0], query)
        q_similarities.append([similarity, document[0], document[1]])

      q_similarities.sort(reverse=True)

      if q_similarities[0][0] >= 0.5:
        #selecting random answer related to similar question
        indices = [i for i in range(len(q_similarities)) if q_similarities[i][1] == q_similarities[0][1]]
        ans = random.choice(indices)
        answer = q_similarities[ans]
        print("BOT:", answer[2])
      else:
        print("BOT: I'm sorry, I am unable to answer this question at the moment.")
      
    # --- QUIT IF USER SAYS bye --- #
    if query == "bye":
      break
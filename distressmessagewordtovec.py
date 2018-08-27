# classify(classifier,'outVec.csv',xfm)
import numpy as np
import pandas as pd
import json
from flask import Flask, request
from sklearn.externals import joblib
import gensim

app = Flask(__name__)

clf = None
clf_m = None
num_features = 100
model = None

@app.route('/')
def hello_world():
    app.run(host='127.0.0.1', port=5000)
    # issue = []
    # issue.append(request.args.get('issue'))
    #
    # pred1 = clf.predict(issue)
    # proba = clf.decision_function(issue)
    #
    # classes = clf.classes_
    # sortedProb = numpy.argsort(-proba[0])
    # pred2 = (classes[sortedProb[1]])
    # pred3 = (classes[sortedProb[2]])
    #
    # print("ML RC 1: ", pred1[0])
    # print("ML RC 2: ", pred2)
    # print("ML RC 3: ", pred3)
    #
    # data = {}
    # data['RC1'] = pred1[0]
    # data['RC2'] = pred2
    # data['RC3'] = pred3
    #
    # json_data = json.dumps(data)
    #
    # return json_data
def vecTransform(X):
    vectors = []
    index2word_set = set(model.wv.index2word)
    for words in X:
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model.wv[word])
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        vectors.append(featureVec)
    return vectors

@app.route('/train', methods=['GET'])
def train():
    global clf
    global clf_m
    global num_features
    num_features = 100
    clf = joblib.load('clfdistress_SVM.pkl', mmap_mode='r+')
    num_features = 100
    # model = gensim.models.Word2Vec.load()
    # Load Google's pre-trained Word2Vec model.
    data = pd.read_csv("textmessages.csv", encoding='latin-1')
    data = data.rename(columns={"v1": "label", "v2": "text"})
    sentences = data["text"]
    # model = gensim.models.Word2Vec(sentences)
    # model = word2vec(sentences)
    # model = word2vec.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in sentences]
    global model
    model = gensim.models.Word2Vec(texts)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    # clf_m = joblib.load('C:/MachineLearning/clf.pkl')
    return "Model Loaded"

@app.route('/isdistress',methods=['POST'])
def hello_world_1():
    #return request
    print request
    
    received_json_data=request.get_json()
    #received_json_data=request.form.get('issue')
    issue = []
    issue.append(received_json_data['issue'].split(" "))
    #issue.append(request.args.get('issue').split(" "))

    pred1 = clf.predict(np.array(vecTransform(issue)))
    # proba = clf_m.decision_function(issue)

    # classes = clf_m.classes_
    # sortedProb = numpy.argsort(-proba[0])
    # pred2 = (classes[sortedProb[1]])
    # pred3 = (classes[sortedProb[2]])
    #
    # print("ML RC 1: ", pred1[0])
    # print("ML RC 2: ", pred2)
    # print("ML RC 3: ", pred3)

    data = {}
    data['isdistress'] = pred1[0]
    # data['RC2'] = pred2
    # data['RC3'] = pred3

    json_data = json.dumps(data)

    return json_data

if __name__ == "__main__":
    app.run(debug=True)


# etree_w2v = Pipeline([
#     ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
#     ("extra trees", MultinomialNB())])
# etree_w2v_tfidf = Pipeline([
#     ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
#     ("extra trees", ExtraTreesClassifier(n_estimators=200))])
# data = pd.read_csv("textmessages.csv",encoding='latin-1')
# #data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
# data = data.rename(columns={"v1":"label", "v2":"text"})
# data.tail()
# data.label.value_counts()
# data['label_num'] = data.label.map({'ham':0, 'help':1})
# data.head()
# sentences = data["text"]
# # model = gensim.models.Word2Vec(sentences)
# # model = word2vec(sentences)
# # model = word2vec.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
# stoplist = set('for a of the and to in'.split())
# texts = [[word for word in document.lower().split() if word not in stoplist]
#          for document in sentences]
#
# #from sklearn.feature_extraction.text import CountVectorizer
# #vect = CountVectorizer()
# #vect.fit(X_train)
# # X = [['Berlin', 'London'],
# #      ['cow', 'cat'],
# #      ['pink', 'yellow']]
# # y = ['capitals', 'animals', 'colors']
# # etree_w2v.fit(X, y)
# #
# # # never before seen words!!!
# test_X = [['help me '], ['I am in trouble'],['leave me alone']]
# etree_w2v.fit(X_train, y_train)
# print(etree_w2v.predict(test_X))

import pickle

def predict(data):
    clf = pickle.load(open("rf_model.pkl",'rb'))
    return clf.predict(data)
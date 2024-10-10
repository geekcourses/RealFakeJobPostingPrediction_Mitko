import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model():
    # loading the model
    model = joblib.load("./models/modelMNB_v1.pkl")
    print(model)
    return model

def load_fit_vectorizer():
    # loading  the data and fitting the vectorizer
    #vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1, 2))
    #df = pd.read_csv("./data/processed/X_train.csv" )
    #X = df['description']
    #vectorizer.fit(X)
    loaded_vectorizer_joblib = joblib.load('./src/tfidf_vectorizer.joblib')
    return loaded_vectorizer_joblib


def user_input_and_predict(model,vectorizer):
    # taking user job description and transforming it into a vector that the model can read and predict if its fraudulent or not

    while True:
        posting = input("Enter a job posting (or 'quit' to exit): ")
        if posting.lower() == "quit":
            break
        
        # the reason its like this is because this is how it works and I don't know why it works this way
        tmp_df = pd.DataFrame([posting], columns=['description'])
        

        posting = vectorizer.transform(tmp_df.iloc[:,0])
        
        prediction = model.predict(posting)
        if prediction == 1:
            print("The posting is fraudulent.")
        else:
            print("The posting is not fraudulent.")

        print("Prediction:", prediction)
        




if __name__ == "__main__":
    model = load_model()
    vectorizer = load_fit_vectorizer()
    user_input_and_predict(model,vectorizer)
    # Enter 'quit' to exit
    
    

    
    



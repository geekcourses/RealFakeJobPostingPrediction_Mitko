import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



def load_model():
    # loading the model
    model = joblib.load("./models/modelMNB_v2.pkl")
    print(model)
    return model

def load_fit_vectorizer():
   # i dont know how to load the vectorizers
    loaded_vectorizer_joblib = joblib.load('./src/tfidf_vectorizer.joblib')
    return loaded_vectorizer_joblib


def user_input_to_dataframe():
    # taking user job input and transforming it into a vector that the model can read and predict if its fraudulent or not
    
    # do I have to make the tranformations that I did to the original dataset?
    

    while True:
        
        tmp_df = {
                        'description': (str(input("Description: "))),
                        'requirements': (str(input("Requirements: "))),
                        'benefits': (str(input("Benefits: "))),
                        }
    
        if tmp_df['description'] == 'quit' or tmp_df['requirements'] == 'quit' or tmp_df['benefits'] == 'quit':
            break
        else:
            return tmp_df
        


def vectorize_and_predict(model, vectorizer):
    posting = vectorizer.transform(tmp_df.iloc[:,0])
        
    prediction = model.predict(posting)

    if prediction == 1:
            print("The posting is fraudulent.")
    else:
            print("The posting is not fraudulent.")

    print("Prediction:", prediction)

        




if __name__ == "__main__":
    tmp_df = user_input_to_dataframe()
    print(tmp_df)
   
    #vectorizer = load_fit_vectorizer()
    #user_input_and_predict(model,vectorizer)
    
    
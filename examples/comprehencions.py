def TfidfVectorizer():
    return 5

features = ['col1', 'col2','col3']

vectorizers = {
    feature: TfidfVectorizer()
        for feature in features
}

print(vectorizers)

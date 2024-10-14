# first line: 1
@memory.cache
def fit_and_transform_tfidf(X_train_raw, X_test_raw, features):
    # Initialize vectorizers for each text feature
    vectorizers = {
        feature: TfidfVectorizer(stop_words='english', min_df=5, max_df=0.8, ngram_range=(1, 2))
        for feature in features
    }

    # List to store TF-IDF matrices
    tfidf_matrices_train = []
    tfidf_matrices_test = []

    # Loop through features to fit and transform
    for feature in features:
        # Fit on training data and transform both train and test data
        vectorizers[feature].fit(X_train_raw[feature])
        tfidf_matrix_train = vectorizers[feature].transform(X_train_raw[feature])
        tfidf_matrix_test = vectorizers[feature].transform(X_test_raw[feature])

        tfidf_matrices_train.append(tfidf_matrix_train)
        tfidf_matrices_test.append(tfidf_matrix_test)

    # Save vectorizers
    joblib.dump(vectorizers, 'tfidf_vectorizers.joblib')
    
    # Combine all TF-IDF matrices into one
    X_train = hstack(tfidf_matrices_train)
    X_test = hstack(tfidf_matrices_test)

    return X_train, X_test

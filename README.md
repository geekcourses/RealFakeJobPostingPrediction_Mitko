# Project Task:
    Create a classification model that uses text data features and meta-features and predict which job description are fraudulent or real.

## Overview:
    The dataset has alot of missing values that had to be filled with placeholders or mode.
    
    The dataset is also imbalanced so oversampling is implamented.

    Trained on MultinomialNB.

    Features: 'description', 'company_profile', 'title', 'requirements'

## Data:
    The dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs..
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from classifiers.metrics_utils import *

def classifier(context):
    classifier_catalog = ['naive bayes', 'linear support vector machine', 'k-nearest neighbors']

    classifier_dict = {
        classifier_catalog[0]:
            {
                'type' : MultinomialNB(alpha=.01),
                'parameters' :  {
                                    'alpha': [0.001, 0.01, 0.1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                                },
                'cv': 10
            },
        classifier_catalog[1]:
            {
                'type': LinearSVC(C=1),
                'parameters': {
                    'C' : np.arange(1,100,10),
                    'tol' : [1e-2, 1e-4, 1e-9]
                }
            },
        classifier_catalog[2]:
            {
            'type': KNeighborsClassifier(n_neighbors=55),
            'parameters': {
                'n_neighbors': np.arange(1,100),
                'weights' : ['uniform', 'distance']
            },
            'cv': 10
        }
    }

    if context.lower() in classifier_dict:

        if context.lower().casefold() == classifier_catalog[0]:
            classifier = classifier_dict.get(context.lower())
            return GridSearchCV(classifier.get('type'), classifier.get('parameters'), cv=classifier.get('cv'),
                                      scoring='accuracy')

        elif context.lower().casefold() == classifier_catalog[1]:
            classifier = classifier_dict.get(context.lower())
            return GridSearchCV(classifier.get('type'), classifier.get('parameters'), cv=classifier.get('cv'),
                                      scoring='accuracy')

        elif context.lower().casefold() == classifier_catalog[2] :
            classifier = classifier_dict.get(context.lower())
            return RandomizedSearchCV(classifier.get('type'), classifier.get('parameters'),cv=classifier.get('cv'), scoring='accuracy', n_iter=10, random_state=5 )

    return None


def evaluate_model(classifier_name, classifier, test_data, predicted):
    from sklearn.metrics import classification_report, confusion_matrix
    print('Overall classification report:')
    print(classification_report(test_data.target, predicted, target_names=test_data.target_names))
    print('Confusion Matrix:')
    print(confusion_matrix(test_data.target, predicted))
    print('Best score for Classifier: '+ classifier_name + ' ' + str(classifier.best_score_))

def text_classification():

    #select a list of categories
    categories = ['alt.atheism', 'comp.graphics', 'sci.space']

    #fetch raw data and split them to train/validate the model
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

    #converting text data into vectors
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train.data)
    test_vectors = vectorizer.transform(test.data)

    #model selection
    nb_classifier = classifier('naive bayes')
    nb_classifier.fit(train_vectors, train.target)
    nb_predicted = nb_classifier.predict(test_vectors)

    svm_classifier = classifier('linear support vector machine')
    svm_classifier.fit(train_vectors, train.target)
    svm_predicted = svm_classifier.predict(test_vectors)

    knn_classifier = classifier('k-nearest neighbors')
    knn_classifier.fit(train_vectors, train.target)
    knn_predicted = knn_classifier.predict(test_vectors)

    classifier_dict = { 'MultinomialNB' : { 'classifier' : nb_classifier, 'predicted' : nb_predicted } ,
                        'KNN' : { 'classifier' : knn_classifier, 'predicted': knn_predicted },
                        'LinearSVC': { 'classifier' : svm_classifier, 'predicted': svm_predicted }
                      }

    #metrics measurements visualization
    display_model_evaluation(train_vectors, train.target, test_vectors, test.target, classifier_dict)


if __name__ == '__main__':
    text_classification()



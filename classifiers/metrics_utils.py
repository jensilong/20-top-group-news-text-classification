from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import metrics

def display_model_evaluation(x_train,y_train,x_test,y_test, clf_dict):

    fig1, axs1 = plt.subplots(3)
    fig1.subplots_adjust(hspace=2, wspace=2)
    fig2, axs2 = plt.subplots(3)
    fig2.subplots_adjust(hspace=2, wspace=2)

    ci = 0
    for classifier_key in clf_dict:
        cm = metrics.confusion_matrix(y_test, clf_dict.get(classifier_key).get('predicted'))
        print('Confusion Matrix of '+ classifier_key  +': ', cm)
        print("Classification Error of " + classifier_key + ": ", 1 - metrics.accuracy_score(y_test, clf_dict.get(classifier_key).get('predicted')))
        print("Sensitivity of " + classifier_key + ": ", metrics.recall_score(y_test, clf_dict.get(classifier_key).get('predicted'), average='weighted'))
        print("Precision of "+ classifier_key + ": ", metrics.precision_score(y_test, clf_dict.get(classifier_key).get('predicted'), average='weighted'))
        print("F-measure of " + classifier_key + ": ", metrics.f1_score(y_test, clf_dict.get(classifier_key).get('predicted'), average='weighted'))

        ROC_multi_class(fig1, axs1, x_train, y_train, x_test, y_test, classifier_key, clf_dict, ci)
        PR_multi_class(fig2, axs2, x_train, y_train, x_test, y_test, classifier_key, clf_dict, ci)
        ci += 1

    plt.show()

def PR_multi_class(fig, axs, x_train, y_train, x_test, y_test, classifier_key, clf_dict, ci):

    classes = [0, 1, 2, 3, 4]
    # Binarize the output
    y_train = label_binarize(y_train, classes=classes)
    n_classes = y_train.shape[1]

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf_dict.get(classifier_key).get('classifier'))
    classifier.fit(x_train, y_train)
    if (clf_dict.get(classifier_key) == clf_dict.get('LinearSVC')):
        y_pred_score = classifier.decision_function(x_test)
    else:
        y_pred_score = classifier.predict_proba(x_test)

    y_test = label_binarize(y_test, classes=classes)

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_pred_score[:, i])

    for i in range(n_classes):
        axs[ci].plot(recall[i], precision[i], label='PR curve of class {0} (area = {1:0.2f})'
                                                ''.format(i + 1, average_precision[i]))

    axs[ci].set_xlim([0.0, 1.0])
    axs[ci].set_ylim([0.0, 1.05])
    axs[ci].set_xlabel('Recall')
    axs[ci].set_ylabel('Precision')
    axs[ci].set_title('Precision-Recall curve of ' + classifier_key + ' multi-class')
    axs[ci].legend(loc="lower right")


def ROC_multi_class(fig, axs, x_train, y_train, x_test, y_test, classifier_key, clf_dict, ci):

    classes = [0, 1, 2, 3, 4]
    # Binarize the output
    ytr = label_binarize(y_train, classes=classes)
    n_classes = ytr.shape[1]

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf_dict.get(classifier_key).get('classifier'))
    classifier.fit(x_train, ytr)
    if (clf_dict.get(classifier_key) == clf_dict.get('LinearSVC')):
        y_pred_score = classifier.decision_function(x_test)
    else:
        y_pred_score = classifier.predict_proba(x_test)

    ytt = label_binarize(y_test, classes=classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytt[:, i], y_pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for the multiclass
    for i in range(n_classes):
        axs[ci].plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i + 1, roc_auc[i]))

    axs[ci].plot([0, 1], [0, 1], 'k--')
    axs[ci].set_xlim([0.0, 1.0])
    axs[ci].set_ylim([0.0, 1.05])
    axs[ci].set_xlabel('False Positive Rate')
    axs[ci].set_ylabel('True Positive Rate')
    axs[ci].set_title('ROC of ' + classifier_key + 'multi-class')
    axs[ci].legend(loc="lower right")



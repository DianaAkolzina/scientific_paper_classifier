from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import numpy as np
def vectorize_data(X):
    """
    Converts text data into a numerical format using TF-IDF.
    Args:
        X (pandas.Series): The text data to be vectorized.
    Returns:
        scipy.sparse.csr.csr_matrix: The vectorized text data.
    """
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)
    print("Shape of TF-IDF Matrix:", X_tfidf.shape)
    print("Type of TF-IDF Matrix:", type(X_tfidf))
    return X_tfidf

def initialize_model():
    """
    Initializes an SVM (Support Vector Machine) model.
    Returns: The initialized SVM model.
    """
    model = SVC()
    return model

def train_svm_model(model, X, y):
    """
    Trains the SVM model on the input data.
    Args:
        model (sklearn.svm.SVC): The SVM model to be trained.
        X_train (scipy.sparse.csr.csr_matrix): The training input data.
        y_train (numpy.ndarray): The training target labels.
    Returns:
        sklearn.svm.SVC: The trained SVM model.
    """
    model.fit(X, y)
    return model
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')

"""
# Example usage
data = get_data_from_gcp(BUCKET_NAME, 'cleaned_data/Updated_df_3000.csv')
X = data['Processed Text']
y = data['Label']

X_tfidf = vectorize_data(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

svm_model = initialize_model()
trained_model = train_svm_model(svm_model, X_train, y_train)
evaluate_model(trained_model, X_test, y_test)
"""

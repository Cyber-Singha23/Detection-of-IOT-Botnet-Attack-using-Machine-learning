import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle


def load_data():
    path = "dataset/"
    benign = pd.read_csv(path + "benign_traffic.csv")

    mirai_scan = pd.read_csv(path + "scan1.csv").head(20000)
    mirai_syn = pd.read_csv(path + "syn.csv").head(20000)
    mirai_ack = pd.read_csv(path + "ack.csv").head(20000)
    mirai_udp = pd.read_csv(path + "udp1.csv").head(20000)
    mirai_udpplain = pd.read_csv(path + "udpplain.csv").head(20000)
    gafgyt_junk = pd.read_csv(path + "junk.csv").head(20000)
    gafgyt_scan = pd.read_csv(path + "scan.csv").head(20000)
    gafgyt_tcp = pd.read_csv(path + "tcp.csv").head(20000)
    gafgyt_udp = pd.read_csv(path + "udp.csv").head(20000)

    malicious_gafgyt_list = [gafgyt_junk, gafgyt_scan, gafgyt_tcp, gafgyt_udp]
    malicious_mirai_list = [mirai_scan, mirai_syn, mirai_ack, mirai_udp, mirai_udpplain]
    malicious_gafgyt_concat = pd.concat(malicious_gafgyt_list)
    malicious_mirai_concat = pd.concat(malicious_mirai_list)

    malicious_mirai_concat['Detection'] = "mirai"
    malicious_gafgyt_concat['Detection'] = "gafgyt"
    benign['Detection'] = "benign"

    combine_data = pd.concat([benign, malicious_mirai_concat, malicious_gafgyt_concat], axis=0)
    combine_data = shuffle(combine_data)

    return combine_data


def preprocess_data(comb_data):
    labels = comb_data.iloc[:, -1]
    labels = np.array(labels).flatten()

    no_labels_data = comb_data.iloc[:, :28]

    scale = StandardScaler()
    scale.fit(no_labels_data)
    scale.transform(no_labels_data)

    X_train, X_test, y_train, y_test = train_test_split(no_labels_data, labels, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test


def DTmodel():
    print('Decision Tree Model:')
    def create_and_train_model():
        decision_tree_classifier = DecisionTreeClassifier()
        return decision_tree_classifier.fit(X_train, np.ravel(y_train))

    def predict_model():
        return decision_tree_model.predict(X_test)

    def evaluate_metrics():
        acc = accuracy_score(y_test, decision_tree_predictions)
        print("\nAccuracy score of decision tree: %.5f" % acc)

        print("\nClassification Report of decision tree:\n",
              classification_report(y_test, decision_tree_predictions, zero_division=0))

    def plot_observation():
        y_test_predict = y_test[:100]
        decision_tree_predict = decision_tree_predictions[:100]
        plt.xlabel('X(Time->)')
        plt.ylabel('0 for Benign Traffic(LOW) and 1 for Malicious Traffic(HIGH)')
        plt.plot(y_test_predict, c='g', label="Benign data")
        plt.plot(decision_tree_predict, c='b', label="Malicious data")
        plt.legend(loc='upper left')
        plt.savefig('Decision Tree.png')

        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(7, 4))
        cm = metrics.confusion_matrix(y_test, decision_tree_predictions, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        ax.set_yticklabels(labels=classes, rotation=0)
        plt.savefig('Confusion_DT.png')

    if __name__ == '__main__':
        comb_data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(comb_data)

        decision_tree_model = create_and_train_model()

        decision_tree_predictions = predict_model()

        evaluate_metrics()

        plot_observation()


def KNNmodel():
    print('K-Nearest Neighbour Model:')
    def create_and_train_model():
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        return knn_classifier.fit(X_train, np.ravel(y_train))

    def predict_model():
        return knn_model.predict(X_test)

    def evaluate_metrics():
        acc = accuracy_score(y_test, knn_predictions)
        print("\nAccuracy score of KNN: %.5f" % acc)
 
        print("\nClassification Report of KNN:\n",
              classification_report(y_test, knn_predictions, zero_division=0))

    def plot_observation():
        y_test_predict = y_test[:100]
        knn_predict = knn_predictions[:100]
        plt.xlabel('X(Time->)')
        plt.ylabel('0 for Benign Traffic(LOW) and 1 for Malicious Traffic(HIGH)')
        plt.plot(y_test_predict, c='g', label="Benign data")
        plt.plot(knn_predict, c='b', label="Malicious data")
        plt.legend(loc='upper left')
        plt.savefig('KNN.png')

        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(7, 4))
        cm = metrics.confusion_matrix(y_test, knn_predictions, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        ax.set_yticklabels(labels=classes, rotation=0)
        plt.savefig('Confusion_KNN.png')

    if __name__ == '__main__':
        comb_data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(comb_data)

        knn_model = create_and_train_model()

        knn_predictions = predict_model()

        evaluate_metrics()

        plot_observation()


def LRmodel():
    print('Logistic Regression Model:')
    def create_and_train_model():
        lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        return lr_classifier.fit(X_train, np.ravel(y_train))

    def predict_model():
        return lr_model.predict(X_test)

    def evaluate_metrics():
        acc = accuracy_score(y_test, lr_predictions)
        print("\nAccuracy score of logistic regression: %.5f" % acc)

        print("\nClassification Report of logistic regression:\n",
              classification_report(y_test, lr_predictions, zero_division=0))

    def plot_observation():
        y_test_predict = y_test[:100]
        lr_predict = lr_predictions[:100]
        plt.xlabel('X(Time->)')
        plt.ylabel('0 for Benign Traffic(LOW) and 1 for Malicious Traffic(HIGH)')
        plt.plot(y_test_predict, c='g', label="Benign data")
        plt.plot(lr_predict, c='b', label="Malicious data")
        plt.legend(loc='upper left')
        plt.savefig('Logistic Regression.png')

        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(7, 4))
        cm = metrics.confusion_matrix(y_test, lr_predictions, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        ax.set_yticklabels(labels=classes, rotation=0)
        plt.savefig('Confusion_LR.png')

    if __name__ == '__main__':
        comb_data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(comb_data)

        lr_model = create_and_train_model()

        lr_predictions = predict_model()

        evaluate_metrics()

        plot_observation()


def RFmodel():
    print('Ramdom Forest Model:')
    def create_and_train_model():
        rf_classifier = RandomForestClassifier()
        return rf_classifier.fit(X_train, np.ravel(y_train))

    def predict_model():
        return rf_model.predict(X_test)

    def evaluate_metrics():
        acc = accuracy_score(y_test, rf_predictions)
        print("\nAccuracy score of random forest: %.5f" % acc)

        print("\nClassification Report of random forest:\n",
              classification_report(y_test, rf_predictions, zero_division=0))

    def plot_observation():
        y_test_predict = y_test[:100]
        rf_predict = rf_predictions[:100]
        plt.xlabel('X(Time->)')
        plt.ylabel('0 for Benign Traffic(LOW) and 1 for Malicious Traffic(HIGH)')
        plt.plot(y_test_predict, c='g', label="Benign data")
        plt.plot(rf_predict, c='b', label="Malicious data")
        plt.legend(loc='upper left')
        plt.savefig('Random Forest.png')

        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(7, 4))
        cm = metrics.confusion_matrix(y_test, rf_predictions, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        ax.set_yticklabels(labels=classes, rotation=0)
        plt.savefig('Confusion_RF.png')

    if __name__ == '__main__':
        comb_data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(comb_data)

        rf_model = create_and_train_model()

        rf_predictions = predict_model()

        evaluate_metrics()

        plot_observation()


def NBmodel():
    print('Naive Bayes Model:')
    def create_and_train_model():
        nb_classifier = GaussianNB()
        return nb_classifier.fit(X_train, np.ravel(y_train))

    def predict_model():
        return nb_model.predict(X_test)

    def evaluate_metrics():
        acc = accuracy_score(y_test, nb_predictions)
        print("\nAccuracy score of naive bayes: %.5f" % acc)

        print("\nClassification Report of naive bayes:\n",
              classification_report(y_test, nb_predictions, zero_division=0))

    def plot_observation():
        y_test_predict = y_test[:100]
        nb_predict = nb_predictions[:100]
        plt.xlabel('X(Time->)')
        plt.ylabel('0 for Benign Traffic(LOW) and 1 for Malicious Traffic(HIGH)')
        plt.plot(y_test_predict, c='g', label="Benign data")
        plt.plot(nb_predict, c='b', label="Malicious data")
        plt.legend(loc='upper left')
        plt.savefig('Naive Bayes.png')

        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(7, 4))
        cm = metrics.confusion_matrix(y_test, nb_predictions, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        ax.set_yticklabels(labels=classes, rotation=0)
        plt.savefig('Confusion_NB.png')

    if __name__ == '__main__':
        comb_data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(comb_data)

        nb_model = create_and_train_model()

        nb_predictions = predict_model()

        evaluate_metrics()

        plot_observation()

        
def MenuSet():
    print("Detection of IOT Botnet using Machine Learning\n")
    print('1. Decision Tree Model')
    print('2. K-Nearest Neighbour Model')
    print('3. Logistic Regression Model')
    print('4. Random Forest Model')
    print('5. Naive Bayes Model')

    try:                                                                
        userInput = int(input("\nSelect An Above Option: "))     
                                                                    
    except ValueError:
        exit("\nThat's Not a Number!!")                                 
        
    else:
        print("\n")                                                     
        
    if (userInput == 1):
        DTmodel()
    elif (userInput == 2):
        KNNmodel()  
    elif (userInput == 3):
        LRmodel()
    elif (userInput == 4):
        RFmodel()
    elif (userInput == 5):
        NBmodel()
    else:
        print("Enter Correct Choice. . . ")
        
MenuSet()

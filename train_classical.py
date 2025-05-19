from sklearn.metrics                    import accuracy_score, classification_report, confusion_matrix
from matplotlib                         import pyplot as plt
import seaborn as sn, joblib
from sklearn.naive_bayes                import MultinomialNB
from sklearn.linear_model               import LogisticRegression
from sklearn.ensemble                   import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from tensorflow.keras.models            import Sequential
from tensorflow.keras.layers            import Dense, Dropout, Activation
from tensorflow.keras.optimizers        import Adam

import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s: %(message)s',
    handlers = [logging.StreamHandler()]
)

def train(model, xv_train, xv_test, y_train, y_test):
    logging.info(f'{model}')

    clf = model()
    clf.fit(xv_train,y_train)

    pred = clf.predict(xv_test)
    accuracy = clf.score(xv_test,y_test)

    report = classification_report(y_test,pred)

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{model}')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')

    logging.info(report)
    logging.info(accuracy)
    # plt.show()
    logging.info('training completed!\n')
    return clf


def neural_network(xv_train, xv_test, y_train, y_test):
    logging.info('neural network')

    model = Sequential([
    Dense(128, activation='relu', input_shape=(10000,)),
    Dropout(rate=0.5),
    Dense(64, activation='relu'),
    Dropout(rate=0.5),
    Dense(1, activation='sigmoid')
])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(xv_train, y_train, epochs=10, batch_size=16, validation_data=(xv_test, y_test))

    loss, accuracy = model.evaluate(xv_test, y_test)
    logging.info(f"Test Accuracy: {accuracy}")

    pred = (model.predict(xv_test) > 0.5).astype("int32")
    logging.info(classification_report(y_test, pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return model


def voting_classifier(xv_train, xv_test, y_train, y_test):
    logging.info('voting classifier')

    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = MultinomialNB()
    clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)], voting='hard')
    clf.fit(xv_train,y_train)

    pred = clf.predict(xv_test)
    accuracy = clf.score(xv_test,y_test)

    report = classification_report(y_test,pred)

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')

    logging.info(report)
    logging.info(accuracy)

    return clf
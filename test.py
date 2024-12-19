import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay

from training import preprocess
from training import RNNTextClassifier, download_dataset


if __name__ == "__main__":
    download_dataset()
    test = pd.read_csv("data/final_test.csv")
    test_features, test_labels = preprocess(test, samples_per_class=5000)

    model = load_model("saved_model.h5")

    rnn = RNNTextClassifier(model=model)
    # rnn.model.load_model('saved_model.h5')

    # predict
    predicted_labels = rnn.predict(test_features)
    actual_labels = test_labels.values.tolist()

    tn, fp, fn, tp = rnn.prediction_metrics(
        y_predicted=predicted_labels, y_actual=test_labels
    )

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ConfusionMatrixDisplay(pd.DataFrame([[tn, fp], [fn, tp]]).to_numpy()).plot()

    print("test accuracy: ", accuracy)
    print("test sensitivity: ", sensitivity)
    print("test specificity: ", specificity)
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def disp_cm(y_true, y_pred, labels=['0', '1']):
    """
    Display confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots()
    ax.set_title('Confusion Matrix')
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)
    plt.show()

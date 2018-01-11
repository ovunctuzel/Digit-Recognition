import cv2
import OCR_CONV
import numpy as np


# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global held
    if event == cv2.EVENT_LBUTTONDOWN:
        held = True
    elif event == cv2.EVENT_LBUTTONUP:
        held = False

    if held:
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)


def clear_img(img):
    cv2.rectangle(img, (0, 0), (512, 336), color=(0, 0, 0), thickness=-1)
    cv2.rectangle(img, (28, 28), (308, 308), color=(200, 200, 200), thickness=2)
    cv2.putText(img, "'C' to clear", (325, 80), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "'Space' to get prediction", (325, 100), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)


def get_prediction(img, model):
    area = img[32:304, 32:304]
    area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    area = cv2.resize(area, (28, 28))
    pred = OCR_CONV.predict(area)
    print "Prediction: ", pred
    cv2.putText(img, "PREDICTION: " + str(pred), (325, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2,
                cv2.LINE_AA)
    return pred


if __name__ == "__main__":
    held = False
    epochs = 1
    for epoch in range(1, epochs + 1):
        OCR_CONV.train(epoch)

    cv2.namedWindow('Digit Recognition')
    cv2.setMouseCallback('Digit Recognition', draw_circle)

    img = np.zeros((336, 512, 3), np.uint8)
    clear_img(img)
    model = 5

    while (True):
        cv2.imshow('Digit Recognition', img)
        key = cv2.waitKey(10)
        if key & 0xFF == 32:
            get_prediction(img, model)
        elif key & 0xFF == 99:
            clear_img(img)
        elif key & 0xFF == 27:
            break

    cv2.destroyAllWindows()

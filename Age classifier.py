import cv2 as cv2
from socket import *
import pickle
import numpy as np
import sklearn

linearModel = pickle.load(open("models/linModel.pkl", "rb")) #rb is write byte mode
sdgModel = pickle.load(open("models/sdgModel.pkl", "rb")) #rb is write byte mode

print("Starting video capture")
cap = cv2.VideoCapture(0)
cap.set(3,640) # adjust width
cap.set(4,480) # adjust height

while True:
    success, img = cap.read()
    
    #resize, rescale and flatten for the model
    new_image = cv2.resize(img, (32, 32)) 
    rows, columns, channels = new_image.shape
    
    new_image = new_image/255
    new_image = new_image.reshape(rows * columns * channels)
    new_image = np.expand_dims(new_image, axis = 0)
    
    linearResult = linearModel.predict(new_image)
    sdgResult = sdgModel.predict(new_image)
    
    img = cv2.putText(img, str(linearResult[0]) , (600,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    img = cv2.putText(img, str(sdgResult[0]) , (0,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    
    cv2.imshow("Webcam", img) # This will open an independent window
    if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cap.release()
        break
        
cv2.destroyAllWindows() 
cv2.waitKey(1) # normally unnecessary, but it fixes a bug on MacOS where the window doesn't close


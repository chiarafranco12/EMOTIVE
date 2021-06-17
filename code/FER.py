#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


Datadirectory = "train/"
Classes = ["0", "1", "2", "3", "4", "5", "6"]


# In[3]:


for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))

        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break


# In[4]:


img_size = 224
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()


# In[5]:


training_Data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


# In[6]:


create_training_Data()


# In[7]:


training_data_length = len(training_Data)
print(training_data_length)


# In[8]:


import random
random.shuffle(training_Data)


# In[9]:


X = []
y = []

for features,label in training_Data:
    X.append(features)
    y.append(label)


# In[10]:


X = np.array(X).reshape(-1, img_size, img_size, 3) #convert to 4 dimension
for i in range(0,training_data_length):
    X[0] = X[0]/255.0; # normalizing


# In[11]:


print(len(X))


# In[12]:


type(y)


# In[13]:


Y = np.array(y)


# In[14]:


Y.shape


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[16]:


model = tf.keras.applications.MobileNetV2()


# In[17]:


model.summary()


# In[18]:


base_input = model.layers[0].input
base_output = model.layers[-2].output


# In[19]:


final_output = layers.Dense(128)(base_output)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_ouput)


# In[20]:


new_model = keras.Model(inputs = base_input, outputs = final_output)
#new_model = tf.keras.models.load_model("Final_mode_95p07.h5")


# In[21]:


new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[ ]:





# In[23]:


new_model.fit(X,Y,epochs=15)


# In[21]:


new_model.save("Final_mode_95p08.h5")


# In[2]:


new_model = tf.keras.models.load_model("Final_mode_95p10.h5")


# In[3]:


new_model.evaluate


# In[ ]:





# In[40]:


frame = cv2.imread("angry.jpeg")


# In[41]:


frame.shape


# In[42]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[43]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[44]:


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# In[45]:


gray.shape


# In[46]:


faces = faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
        roi_gray =  gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else: 
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]


# In[47]:


final_image = cv2.resize(face_roi, (224,224))
final_image = np.expand_dims(final_image, axis = 0)
final_image = final_image/255.0


# In[48]:


Predictions = new_model.predict(final_image)


# In[49]:


Predictions[0]


# In[50]:


np.max(Predictions[0])


# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
new_model = tf.keras.models.load_model("Final_mode_95p10.h5")


# In[3]:


#VIDEO DETECTION
import cv2

path = "haarcascade_frontalface_default.xml" #
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

rectangle_bgr = (255, 255, 255)
img = np.zeros((500, 500))
text = "Some text in a box!"
(text_width, text_height) = cv2.getTextSize(text, font, fontScale = font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
face_roi = [] 
while  True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray =  gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 3)
        if len(facess) == 0:
            print("Face not detected")
        else: 
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]
    
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    Predictions = new_model.predict(final_image)
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
#1:16:29
    
    if  (np.argmax(Predictions) == 0):
        status = "Angry"
    elif (np.argmax(Predictions) == 1):
        status = "Disgust"
    elif (np.argmax(Predictions) == 2):
        status = "Fear"
    elif (np.argmax(Predictions) == 3):
        status = "Happy"
    elif (np.argmax(Predictions) == 4):
        status = "Sad"
    elif (np.argmax(Predictions) == 5):
        status = "Surprise"
    elif (np.argmax(Predictions) == 6):
        status = "Neutral"
    else:
        status = "not detected"
        
    x1,y1,w1,h1 = 0,0,175,75
    cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0), -1)
    cv2.putText(frame, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255))
    
    print(status)

    cv2.imshow('Face Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[5]:


#VIDEO DETECTION PROVA
import face_recognition
import cv2 # pip install opencv-python
import os

 
webcam = cv2.VideoCapture(0)

#creazione lista nomi file da confrontare con video
arr = os.listdir()
fotoDB = []
for x in arr:
    if '.' in x:
        estensione = x.split('.')[1]
    else:
        estensione = ''
    if estensione == 'jpg':
        fotoDB.append(x)

print(fotoDB)

for image_file in fotoDB:
    count = 0
    label = ''
    #image_file = input("Target Image File > ")
    target_image = face_recognition.load_image_file(image_file)
    target_encoding = face_recognition.face_encodings(target_image)[0]
    target_name = image_file.split('.')[0]
    print("Image Loaded. 128-dimensions Face Encoding Generated " + target_name)

    process_this_frame = True

    while count < 10:
        ret, frame = webcam.read()
        small_frame = cv2.resize(frame, None, fx=0.20, fy=0.20)
        rgb_small_frame = cv2.cvtColor(small_frame, 4)

        if process_this_frame:
            face_location = face_recognition.face_locations(rgb_small_frame)
            frame_encodings = face_recognition.face_encodings(rgb_small_frame)

            if frame_encodings:
                frame_face_encoding = frame_encodings[0]
                match = face_recognition.compare_faces([target_encoding],frame_face_encoding)[0]
                label = target_name if match else "Unknown"
        
        process_this_frame = not process_this_frame

        if face_location:
            top, right, bottom, left = face_location[0]

            top *= 5
            right *= 5
            bottom *= 5
            left *= 5 

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0,255,0), cv2.FILLED)
            label_font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), label_font, 0.8, (255,255,255), 1)
        
        cv2.imshow("Video Feed", frame)
        if (label == "Unknown"):
            count += 1
        else:
            count = 0

        print('label: ' + label + '\n' + 'count: ' + str(count))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





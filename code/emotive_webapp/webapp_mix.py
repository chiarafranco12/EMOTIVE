# https://github.com/giulioangiani/programming/tree/master/WEB_Programming
#Model View Controller
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response, session # importa le librerie del framework
from flask_session import Session
# Database
import mysql.connector
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dbconn import * # connessione al DataBase

# gestione file e cartelle
from werkzeug.utils import secure_filename   
import os
import shutil # spostare file da una cartella ad un'altra 

# analisi video
import cv2

import matplotlib.pyplot as plt

# array numpy
import numpy as np

# algoritmo di riconoscimento emozioni e identità
import tensorflow as tf
import face_recognition

import json

from datetime import timedelta

# variabili globali
UPLOAD_FOLDER = 'static/uploads/'
SAVE_FOLDER  = 'static/save'
UPLOAD_FOLDER_FACES = 'static/faces/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
last_emotion = "niente"
close = True
camera = False
target_name = ''
target_encoding = ''
count = 0
label = ''
#session = []

# configurazione delle variabili
app = Flask(__name__) # crea la web app
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_FACES'] = UPLOAD_FOLDER_FACES
app.config['SAVE_FOLDER'] = SAVE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'super secret'
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_TYPE"] = "filesystem"
app.permanent_session_lifetime = timedelta(hours=5) # durata della sessione
#session = []

# algoritmo di riconoscimento delle emozioni
new_model = tf.keras.models.load_model("Final_mode_95p10.h5")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #identificazione faccia

def session_check(): # controlla se l’utente si è già loggato e la sessione è attiva
    if 'username' not in session:
        print("not session")
        return False
    else:
        print("yes session")
        return True

def allowed_file(filename): # controlla se la foto inserita ha un’estensione compresa tra quelle consentite
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def close_fun(request_form): # apre o chiude la telecamera in funzione del pulsante premuto dall’utente
    global camera
    global close
    if camera:
        if 'stop' in request_form:
            camera.release()
            cv2.destroyAllWindows()
            close = True
    if 'start' in request_form:
        close = False
        video_feed()

@app.route('/image_classifier')
def image_classifier(frame): # identifica il viso principale presente nel frame passato come parametro e l'emozione da esso espressa
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    Predictions = []
    face_roi = []
    x, y, w, h = 0, 0, 0, 0
    for x,y,w,h in faces:
            roi_gray =  gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            facess = faceCascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 4)
            if len(facess) == 0:
                print("Face not detected")
            else: 
                for (ex,ey,ew,eh) in facess:
                    face_roi = roi_color[ey: ey+eh, ex:ex + ew]        

    if len(face_roi) > 0:
        final_image = cv2.resize(face_roi, (224,224))
        final_image = np.expand_dims(final_image, axis = 0)
        final_image = final_image/255.0
        Predictions = new_model.predict(final_image)
    else:
        Predictions.insert(0,-1)

    #riconoscimento emozione
    if (np.argmax(Predictions) == 0):
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
        status = "Not Defined"

    global last_emotion
    last_emotion = status
    '''
    if session_check():
        session['emotion'] = last_emotion
    '''
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    x1,y1,w1,h1 = 0,0,175,75
    cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0), -1)
    cv2.putText(frame, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    '''
    return frame

def recognize(frame): # identifica la persona presente nel frame passato come parametro
    global target_name
    global target_encoding
    global count
    global label

    small_frame = cv2.resize(frame, None, fx=0.20, fy=0.20)
    rgb_small_frame = cv2.cvtColor(small_frame, 4)

    face_location = face_recognition.face_locations(rgb_small_frame)
    frame_encodings = face_recognition.face_encodings(rgb_small_frame)

    if frame_encodings:
        frame_face_encoding = frame_encodings[0]
        match = face_recognition.compare_faces([target_encoding],frame_face_encoding)[0]
        label = target_name if match else "Unknown"


    if face_location:
        top, right, bottom, left = face_location[0]

        top *= 5
        right *= 5
        bottom *= 5
        left *= 5 
        
        cv2.rectangle(frame, (left, top), (right, bottom), (235, 99, 37), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (235, 99, 37), cv2.FILLED)
        label_font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), label_font, 0.8, (255,255,255), 1)

        global last_emotion
        label2 = last_emotion
        cv2.rectangle(frame, (left, bottom), (right, bottom + 30), (235, 99, 37), cv2.FILLED)
        cv2.putText(frame, label2, (left + 6, bottom + 24), label_font, 0.8, (255,255,255), 1)

    if (label == "Unknown"):
        count += 1
    else:
        count = 0

    return frame

def gen_frames(): # ciclo while che genera i frame uno per uno e li unisce per creare il video
    global target_name
    global target_encoding
    global count
    global label
    global last_emotion

    if not close:
        global camera 
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise IOError("Cannot open webcam")
        
        while True:
            sql = "SELECT path_f, name_f, surname_f FROM Face;"
            faces = mysql_query(sql,"")
            print(faces)
            for face in faces:
                count = 0
                label = ''
                filename = UPLOAD_FOLDER_FACES + face['path_f']
                name = face['name_f']
                surname = face['surname_f']
                
                target_image = face_recognition.load_image_file(filename)
                target_encoding = face_recognition.face_encodings(target_image)[0]
                target_name = name + ' ' + surname
                print("Image Loaded. 128-dimensions Face Encoding Generated " + target_name)

                while count < 10:# Capture frame-by-frame
                    success, frame = camera.read()  # read the camera frame
                    frame = image_classifier(frame)
                    frame = recognize(frame)

                    # se il programma ha ricnosciuto il viso ed è già presente nel DB gli assegna nome e cognome corretti
                    if label != 'Unknown':
                        name = label.split()[0]
                        surname = label.split()[1]
                    # altrimenti se la faccia non viene riconosciuta vengono generati in modo sequenziale
                    else:
                        sql = "SELECT MAX(id) as max FROM Face;"
                        maxid = mysql_query(sql,"")

                        name = 'utente' + str(maxid[0]['max'] + 1)
                        surname = 'utente' + str(maxid[0]['max'] + 1)
                    var = (filename, name, surname, 'guser_id', last_emotion)

                    print(last_emotion)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  #concatenazione e visualizzazione dei frame

@app.route('/')
def home(): # pagina home all’idirizzo ‘/‘
    return render_template('index.html',**vars())

@app.route('/nav')
def nav(): # pagina principale contenente il menù di navigazione all’idirizzo ‘/nav‘
    return render_template('nav.html',**vars())

@app.route('/download')
def download(): # pagina download all’idirizzo ‘/download‘
    return render_template('download.html',**vars())

@app.route('/project')
def project(): # pagina project all’idirizzo ‘/project‘
    return render_template('project.html',**vars())

@app.route('/contact')
def contact(): # pagina contact all’idirizzo ‘/contact‘
    return render_template('contact.html',**vars())

@app.route('/settings')
def settings(): # pagina settings all’idirizzo ‘/settings‘
    return render_template('settings.html',**vars())




# Video Analyze and Show
@app.route('/video_capture')
def video_capture(): # restituisce la pagina per avviare l’analisi del video
    global last_emotion
    global close

    if not session_check():
        return redirect('/login')
    else:
        return render_template('video_capture.html',**vars())

@app.route('/video_capture', methods=['POST'])
def open_close_camera(): # chiama la funzione ‘close_fun()’ e poi restituisce la pagina per avviare l’analisi del video
    global close

    if not session_check():
        return redirect('/login')

    close_fun(request.form)
    return render_template('video_capture.html',**vars())

@app.route('/video_feed')
def video_feed(): # chiama la funzione ‘gen_frames()’
    if not session_check():
        return redirect('/login')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

'''
# Image Upload and Show
@app.route('/display_image')
def display_image(): # restituisce la pagina che permette di inserire un’immagine e mostrarla sullo schermo
    if not session_check():
        return redirect('/login')
    else:
        return render_template("display_image.html",**vars())

@app.route('/display_image', methods=['POST'])
def upload_file(): # riceve il file caricato dall’utente, lo analizza e lo salva nella cartella ‘uploads’
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        frame = cv2.imread(UPLOAD_FOLDER + filename) # leggi file salvato
        frame = image_classifier(frame) # analizza l'immagine: trova emozione

        global target_name
        global target_encoding
        global count
        global label

        sql = "SELECT path_f, name_f FROM Face;"
        faces = mysql_query(sql,"")
        print(faces)

        for face in faces: # controlla se la faccia inserita è presente nel DB
            count = 0
            label = ''
            filename = UPLOAD_FOLDER_FACES + face['path_f']
            name = face['name_f']
            
            target_image = face_recognition.load_image_file(filename)
            target_encoding = face_recognition.face_encodings(target_image)[0]
            target_name = name

            analized_image = recognize(frame) # analizza immagine: riconosce

            if label != 'Unknown':
                break
        filename = face['path_f']
        cv2.imwrite(os.path.join(app.config['SAVE_FOLDER'], filename), analized_image) # salva immagine analizzata nella cartella

        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('display_image.html', **vars())
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display(filename): # restituisce l’immagine appena caricata dall’utente dalla pagina ‘display_image’
    return redirect(url_for('static', filename='save/' + filename), code=301) #per visualizzare un immagine salvata su pc
'''
# Login
@app.route('/login')
def login_page(): # elimina i dati presenti nella session e restituisce la pagina ‘login.html’
    session.clear()
    return render_template("login.html",**vars())

@app.route('/login', methods=['POST'])
def login_check(): # controlla se i dati inseriti dall’utente sono corretti, in tal caso lo reindirizza alla home, altrimenti fa ritentare il login
    if 'username' in session:
        return redirect("/nav")

    session.permanent = True

    sql = "SELECT id, username_u FROM GUser WHERE username_u = %s AND password_u = %s ;"
    error = None
    if 'username_email' in request.form and 'password' in request.form:
        username_email = request.form['username_email']
        password = request.form['password']
    else:
        username_email = ""
        password = ""

    var = (username_email, password)
    utente = mysql_query(sql,var)

    print(utente)   
    if len(utente) > 0:
        utente = utente[0]
        session['id'] = utente['id']
        session['username'] = utente['username_u']
    else:
        sql = "SELECT g.id as id, g.username_u as username_u FROM GUser g, NUser u WHERE u.email_u = %s AND g.password_u = %s AND g.id = u.guser_id"
        utente = mysql_query(sql,var)
        if len(utente) > 0:
            utente = utente[0]
            session['id'] = utente['id']
            session['username'] = utente['username_u']
        else:
            error = 'Invalid username/password'
            return render_template('login.html', **vars())
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return redirect('/nav')

# Logout
@app.route('/logout')
def logout(): # elimina i dati presenti nella session e restituisce la pagina ‘login.html’
    session.clear()
    return redirect("/nav")

# Sing Up
@app.route('/signup')
def signup_page(): # elimina i dati presenti nella session e restituisce la pagina ‘signup.html’ per registrare un nuovo utente
    session.clear()
    return render_template("signup.html",**vars())

@app.route('/signup', methods=['POST'])
def signup_insert(): # controlla se i dati inseriti corrispondono ad un utente già esistente, in caso contrario crea il nuovo utente inserendolo nel DB
    if 'username' in session:
        return redirect("/nav")
    
    #controlla se l'utente è già presente nel DB
    username = request.form['u']
    var = (username)
    sql = "SELECT id FROM GUser WHERE username_u = %s;"
    utente = mysql_query(sql, var)
    if len(utente) > 0:
        user_exist = True
        return render_template("signup.html",**vars())

    # altrimenti lo inserisce
    name = request.form['n']
    surname = request.form['s']
    birthdate = request.form['d']
    email = request.form['e']
    email_conferma = request.form['e2']

    if email != email_conferma:
        email_check = False

    reason = request.form['r']
    password = request.form['p']
    password_conferma = request.form['p2']

    if password != password_conferma:
        password_check = False
    
    if not email_check or not password_check:
        return render_template("signup.html",**vars())

    var = (name, surname, username, password)
    sql = "INSERT INTO GUser (name_u, surname_u, username_u, password_u) VALUES (%s,%s,%s,%s)"
    mysql_insert(sql, var)
    
    
    var = ""
    sql = "SELECT MAX(id) as id FROM GUser"
    lastid = mysql_query(sql, "")
    lastid = lastid[0]['id']

    var = (birthdate, email, reason, lastid)
    sql = "INSERT INTO GUser (birthdate, email, reason, guser_id) VALUES (%s,%s,%s,%s)"
    mysql_insert(sql, var)

    return redirect('/home')

# upload foto per DB
@app.route('/analyze_face')
def analyze_page():
    if not session_check():
        return redirect('/login')
    return render_template('analyze_face.html', **vars())

@app.route('/analyze_face', methods=['POST'])
def analyze_face(): # riceve il file caricato dall’utente, lo salva nella cartella ‘faces’ e inserisce nel DB i dati relativi all'immagine
    if not session_check():
        return redirect('/login')

    if 'file' in request.files:
        name = ""
        surname = ""
        file = request.files['file']
        guser_id = session['id']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("filename: " + filename)

            # salva l'immagine nella cartella degli upload
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            frame = cv2.imread(UPLOAD_FOLDER + filename) # leggi file salvato
            frame = image_classifier(frame) # analizza l'immagine: trova emozione

            global target_name
            global target_encoding
            global count
            global label

            # seleziona tutte le facce presenti nel database per effettuare il riconoscimento facciale
            sql = "SELECT path_f, name_f, surname_f FROM Face;"
            faces = mysql_query(sql,"")
            fine = False

            analized_image = frame
            for face in faces: # controlla se la faccia inserita è presente nel DB
                if not fine:
                    count = 0
                    label = 'Unknown'
                    name_try = face['name_f']
                    surname_try = face['surname_f']
                    filename_try = UPLOAD_FOLDER_FACES + face['path_f']
                    
                    target_image = face_recognition.load_image_file(filename_try)
                    target_encoding = face_recognition.face_encodings(target_image)[0]
                    target_name = name_try + ' ' + surname_try

                    analized_image = frame
                    analized_image = recognize(analized_image) # analizza immagine: riconosce

                if label != 'Unknown':
                    fine = True

            # se il programma ha ricnosciuto il viso ed è già presente nel DB gli assegna nome e cognome corretti
            if label != 'Unknown':
                print(label.split())
                name = label.split()[0]
                surname = label.split()[1]
            # altrimenti se la faccia non viene riconosciuta vengono generati in modo sequenziale
            else:
                sql = "SELECT MAX(id) as max FROM Face;"
                maxid = mysql_query(sql,"")

                name = 'utente' + str(maxid[0]['max'] + 1)
                surname = 'utente' + str(maxid[0]['max'] + 1)
                unknown = True
            
            # seleziona l'id dell'emozione 
            sql = f"SELECT id FROM Emotion WHERE description_e = '{last_emotion}'"
            emotion_id = mysql_query(sql, "")
            emotion_id = emotion_id[0]['id']

            var = (filename, name, surname, guser_id, emotion_id)

            sql = "INSERT INTO Face (path_f, name_f, surname_f, fk_guser_id, fk_emotion_id) VALUES (%s, %s, %s, %s, %s)"
            
            mysql_insert(sql, var)

            # sposta il file dalla cartella 'upload' alla cartella 'faces'
            shutil.move(UPLOAD_FOLDER + filename, UPLOAD_FOLDER_FACES + filename)

            # salva nella cartella 'save' la foto analizzata
            cv2.imwrite(os.path.join(SAVE_FOLDER, filename), analized_image) # salva immagine analizzata nella cartella

            session['var'] = list(var)
            session['emotion'] = last_emotion
            print("session: ")
            print(session)

            print('upload_image filename: ' + filename)
            flash('Image successfully uploaded and displayed below')
            return render_template('analyze_face.html', **vars())
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        flash('No file part')
        return redirect(request.url)

# upload foto per DB
@app.route('/upload_face')
def upload_page():
    if not session_check():
        return redirect('/login')
    return render_template('upload_face.html', **vars())

@app.route('/upload_face', methods=['POST'])
def upload_face(): # riceve il file caricato dall’utente, lo salva nella cartella ‘faces’ e inserisce nel DB i dati relativi all'immagine
    if not session_check():
        return redirect('/login')

    if 'file' in request.files:
        name = ""
        surname = ""
        emotion = 0
        if 'name' in request.form and 'surname' in request.form:
            name = request.form['name']
            surname = request.form['surname']

        if 'emotion' in request.form:
            emotion = request.form['emotion']

        name_check = True
        surname_check = True
        emotion_check = True
        if emotion == 0:
            emotion_check = False
        if name == '':
            name_check = False
        if surname == '':
            surname_check = False
        if not emotion_check or not surname_check or not name_check:
            return redirect('/upload_face')

        file = request.files['file']
        guser_id = session['id']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("filename: " + filename)
            # prova a trovare nel DB una riga che corrisponda ai dati inseriti
            var = (filename, name, surname, guser_id, emotion)
            sql = "SELECT id FROM Face WHERE path_f = %s AND name_f = %s AND surname_f = %s AND fk_guser_id = %s AND fk_emotion_id = %s;"
            face = mysql_query(sql, var)
            
            if len(face) > 0:
                # foto già esistente
                return redirect('/upload_face')
            else:
                # salva l'immagine nella cartella degli upload
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            frame = cv2.imread(UPLOAD_FOLDER + filename) # leggi file salvato

            var = (filename, name, surname, guser_id, emotion)
            sql = "INSERT INTO Face (path_f, name_f, surname_f, fk_guser_id, fk_emotion_id) VALUES (%s, %s, %s, %s, %s)"
            mysql_insert(sql, var)

            # sposta il file dalla cartella 'upload' alla cartella 'faces'
            shutil.move(UPLOAD_FOLDER + filename, UPLOAD_FOLDER_FACES + filename)


            session['var'] = list(var)
            print("session: ")
            print(session)

            print('upload_image filename: ' + filename)
            flash('Image successfully uploaded and displayed below')
            return redirect('/people')
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        flash('No file part')
        return redirect(request.url)


@app.route('/check_result', methods=['POST'])
def check_result(): # chiede all’utente se i dati rilevati sto corretti, altrimenti lo reindirizza sulla pagina in cui potrà inserire i dati corretti
    if not session_check():
        return redirect('/login')

    print(request.form)
    if 'right' in request.form:
        wrong = False
    elif 'wrong' in request.form:
        wrong = True

    return render_template('correct.html', **vars())

@app.route('/correct',  methods=['POST'])
def correct(): 
    if not session_check():
        return redirect('/login')

    if 'name' in request.form and 'surname' in request.form and 'emotion' in request.form:
        name = str(request.form['name'])
        surname = str(request.form['surname'])
        emotion = str(request.form['emotion'])
        guser_id = session['id']

        # <SecureCookieSession {'_permanent': True, 'id': 1, 'username': 'chiaretta02', 
        # 'var': ['chiara2.jpg', 'Chiara', 'Franco', 1, 'Happy']}>
        # seleziona l'id dell'ultima immagine inserita dall'utente corrente

        var = session['var']
        sql = "SELECT id FROM Face WHERE path_f = %s AND name_f = %s AND surname_f = %s AND fk_guser_id = %s AND fk_emotion_id = %s;"
        face_id = mysql_query(sql, var)

        print("var:")
        print(var)

        if name == '':
            name = var[1]
        if surname == '':
            surname = var[2]
        if emotion == '':
            emotion = var[4]

        face_id = face_id[0]['id']
        newvar = (name, surname, emotion, guser_id, face_id)
        
        # aggiorna  il DB con i dati corretti
        sql = "UPDATE Face SET name_f = %s, surname_f = %s, fk_emotion_id = %s WHERE fk_guser_id = %s AND id = %s;"
        mysql_insert(sql, newvar)

        newvar = (name, surname, guser_id)

        # elimina la versione precedente senza emozione
        sql = "DELETE FROM Face WHERE name_f = %s AND surname_f = %s AND fk_emotion_id IS NULL AND fk_guser_id = %s;"
        mysql_insert(sql, newvar)

        wrong = False

        return render_template('correct.html', **vars())

# Show DataBase
@app.route('/people')
def people(): # seleziona dal DB le facce inserite dall’utente attualmente loggato e le mostra restituendo la pagina ‘db.html’ 
    if not session_check():
        return redirect('/login')
    else:
        name_filter = "%"
        surname_filter = "%"
        emotion_filter = 0
        if 'name_filter' in request.form:
            name_filter = '%' + request.form['name_filter'] + '%'
        if 'surname_filter' in request.form:
            surname_filter = '%' + request.form['surname_filter'] + '%'
        if 'emotion_filter' in request.form:
            emotion_filter = request.form['emotion_filter']

        guser_id = session['id']
        var = (guser_id, name_filter, surname_filter, emotion_filter)

        # seleziona l'id dell'emozione
        if emotion_filter == 0:
            var_temp = (guser_id, name_filter, surname_filter)
            sql = "SELECT * FROM Face WHERE fk_guser_id = %s AND name_f LIKE %s AND surname_f LIKE %s;"
        else:
            var_temp = var
            sql = "SELECT * FROM Face WHERE fk_guser_id = %s AND name_f LIKE %s AND surname_f LIKE %s AND fk_emotion_id = %s;"
            
        face = mysql_query(sql, var_temp)
        print(face)

        return render_template('db.html', **vars())

@app.route('/people', methods=['POST'])
def people_filter():
    if not session_check():
        return redirect('/login')

    name_filter = "%"
    surname_filter = "%"
    emotion_filter = "%"
    if 'name_filter' in request.form and request.form['name_filter'] != '':
        name_filter = '%' + request.form['name_filter'] + '%'
    if 'surname_filter' in request.form and request.form['surname_filter'] != '':
        surname_filter = '%' + request.form['surname_filter'] + '%'
    if 'emotion_filter' in request.form:
        emotion_filter = request.form['emotion_filter']

    guser_id = session['id']
    var = (guser_id, name_filter, surname_filter, emotion_filter)

    # seleziona l'id dell'emozione
    if emotion_filter == '0':
        var_temp = (guser_id, name_filter, surname_filter)
        sql = "SELECT * FROM Face WHERE fk_guser_id = %s AND name_f LIKE %s AND surname_f LIKE %s;"
    else:
        var_temp = var
        sql = "SELECT * FROM Face WHERE fk_guser_id = %s AND name_f LIKE %s AND surname_f LIKE %s AND fk_emotion_id = %s;"
        
    face = mysql_query(sql, var_temp)
    print(face)

    return render_template('db.html', **vars())

@app.route('/user_profile')
def user_page():
    if not session_check():
        return redirect('/login')

    user_id = session["id"] 
    sql = f"SELECT g.name_u as name, g.surname_u as surname, g.username_u as username, rr.description_r as role FROM GUser g, Role_u rr WHERE g.id = {user_id} AND g.role_id = rr.id"
    guser_data = mysql_query(sql,"")
    sql = f"SELECT u.birthdate_u as birthdate, u.email_u as email, r.description_m as reason FROM GUser g, NUser u, Reason r WHERE g.id = {user_id} AND u.guser_id = {user_id} AND u.reason_id = r.id"
    nuser_data = mysql_query(sql,"")

    print (guser_data)
    print(nuser_data)
    guser_data = guser_data[0]
    if len(nuser_data) > 0:
        nuser_data = nuser_data[0]
    return render_template('profile.html', **vars())


if __name__ == '__main__':
    app.secret_key = "mysecretkey"
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(
        host="0.0.0.0",         # host
        port=9000,              # port
        debug=True,             # auto debug active
    )


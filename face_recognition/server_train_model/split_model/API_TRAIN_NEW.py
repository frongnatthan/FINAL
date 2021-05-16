import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import os
import os.path
import cv2
import math
from sklearn import neighbors
import pickle
from PIL import Image, ImageDraw

from flask import Flask,send_from_directory,jsonify
from flask_restful import Api,Resource,reqparse
import werkzeug, os
import shutil

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
process_stat=False
train_stat=False

app=Flask(__name__)
api=Api(app)
path=os.getcwd()
UPLOAD_FOLDER = path+'/VideoForPic'
app.config["CLIENT_FILE"] = path
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')
parser.add_argument('name', type=str, location='form')


def train(train_dir, model_save_path, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    print("Training KNN classifier...")
    X = []
    y = []
    i = 1
    
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        print(type(class_dir))
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        n=len(os.listdir(os.path.join(train_dir, class_dir)))


        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image,model="cnn")

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
            print("Img : "+str(i)+" from "+str(n))
            i += 1
        i=1
        print("folder : "+str(class_dir)+" finished")
        shutil.rmtree("PicForTrain/"+class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)
    return knn_clf
    #return send_from_directory(app.config["CLIENT_FILE"], model_save_path, as_attachment=True)

def create_dataset(img,id,img_id,nameFolder):
        cv2.imwrite("PicForTrain/"+nameFolder+"/"+nameFolder+"."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
        features=classifier.detectMultiScale(img,scaleFactor,minNeighbors)
        print(features)
        coords=[]
        for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                
        return img
        
def detect(img,faceCascade,img_id,nameFolder):        
        id = 1
        create_dataset(img,id,img_id,nameFolder)
        return img


def process(filename):
    shutil.rmtree("models")
    os.mkdir("models")
    print("Start Crop Picture!")
    videoFile = filename
    print("VideoForPic"+"/"+videoFile)
    a=videoFile.replace('.mp4','')
    print(a)
    os.mkdir('PicForTrain/'+a)
    img_id=0        
    cap = cv2.VideoCapture("VideoForPic"+"/"+videoFile)
    #cap = cv2.VideoCapture('rtsp://admin:kusrc12345@158.108.122.7:554/stream')
    while (True):
        ret,frame = cap.read()
        if(ret is False):
            print("break")
            break
        print(ret)
        frame=detect(frame,faceCascade,img_id,a)
        img_id+=1
    cap.release()
    cv2.destroyAllWindows()
    os.remove("VideoForPic/"+videoFile)
    print("Cropping complete!")


class Upload(Resource):
    decorators=[]
    def post(self):
        data = parser.parse_args()
        print(data)
        photo = data['file']
        print(photo)
        name= data['name']
        filename = name+'.mp4'
        photo.save(os.path.join(UPLOAD_FOLDER,filename))
        process_stat=process(filename)
        file_model=name+'.clf'
        return train("PicForTrain","models/"+file_model, n_neighbors=2)


api.add_resource(Upload,"/upload")

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=9997, threaded = True ,debug=False)

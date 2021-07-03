from tkinter import filedialog,ttk
from tkinter import Button, Label

from tkinter import *

import cv2
import shutil,os
from PIL import ImageTk,Image
from functools import partial
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle


master = Tk()

master.geometry('400x500')

def change(root,filename):
    root.destroy()
    newwin2(filename)

def change2(root2,filename):
    root2.destroy()
    newwin3(filename)
    
def newwin(filename):
    master.destroy()
    root=Tk()
    canvas2=Canvas(root,width=400,height= 500)
    canvas2.pack()

    img=Image.open(filename)
    img=img.resize((320,300),Image.ANTIALIAS)
    img=ImageTk.PhotoImage(img)
    canvas2.create_image(40,40,anchor=NW,image=img)

    capture1 = Button(root,text="capture")
    capture1.place(x=260,y=380)

    select1 =Label(root,text="Selected Image",font=("Courier", 15))
    select1.place(x=120,y=20)

    browse1 = Button(root,text="Browse",command=browseImage)
    browse1.place(x=60,y=380)

    nextb1 = Button(root,text="next",command=partial(change,root,filename))
    nextb1.place(x=180,y=360)
    root.mainloop()
    

def newwin4(filename,label):
    root3=Tk()
    canvas1=Canvas(root3,width=526,height= 500)
    canvas1.pack()

    im = Image.open(filename)
    im= im.resize((320, 300), Image.ANTIALIAS)
    im= ImageTk.PhotoImage(im)
    canvas1.create_image(40, 40, anchor=NW, image=im)
    lbl= Label(canvas1,text='Predicted Class:{}'.format(label))
    lbl.config(font=("Courier", 15))
    lbl.place(x=40,y=380)

    root3.mainloop()


def newwin3(filename):
    image = cv2.imread(filename)
    output = image.copy()
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    print("[INFO] loading network...")
    model = load_model(r"C:\Users\Dipesh\Desktop\Leaf\model10.model")
    lb = pickle.loads(open(r"C:\Users\Dipesh\Desktop\Leaf\model10.pickle", "rb").read())
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    #filename1 = args["image"][args["image"].rfind(os.path.sep) + 1:]
    #correct = " " if filename1.rfind(label) != -1 else " "
    label = "{} ".format(label)
    print("[INFO] {}".format(label))
    newwin4(filename,label)
    

def newwin2(filename):
    root2 = Tk()  
    canvas = Canvas(root2, width = 1100, height = 750)  
    canvas.pack()  
    img = Image.open(filename)
    img=img.rotate(45)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img= ImageTk.PhotoImage(img)
    canvas.create_image(40, 40, anchor=NW, image=img)
    lb= Label(canvas,text='Rotated at 45 :')
    lb.config(font=("Courier", 15))
    lb.place(x=40,y=20)

    img1= Image.open(filename)
    img1=img1.rotate(25)
    img1 = img1.resize((300, 300), Image.ANTIALIAS)
    img1= ImageTk.PhotoImage(img1)
    canvas.create_image(40, 400, anchor=NW, image=img1)
    lb= Label(canvas,text='Rotated at 25 : ')
    lb.config(font=("Courier", 15))
    lb.place(x=40,y=380)

    img2= Image.open(filename)
    img2 = img2.resize((300, 300), Image.ANTIALIAS)
    img2=img2.rotate(90)
    img2= ImageTk.PhotoImage(img2)
    canvas.create_image(400, 40, anchor=NW, image=img2)
    lb= Label(canvas,text='Roatated at 90 :')
    lb.config(font=("Courier", 15))
    lb.place(x=400,y=20)

    img3= Image.open(filename)
    img3 = img3.resize((300, 300), Image.ANTIALIAS)
    img3=img3.rotate(180)
    img3= ImageTk.PhotoImage(img3)
    canvas.create_image(400, 400, anchor=NW, image=img3)
    lb= Label(canvas,text='Vertical Flip :')
    lb.config(font=("Courier", 15))
    lb.place(x=400,y=380)

    img4= Image.open(filename)
    img4 = img4.resize((300, 300), Image.ANTIALIAS)
    img4= img4.transpose(Image.FLIP_LEFT_RIGHT)
    img4= ImageTk.PhotoImage(img4)
    canvas.create_image(760, 40, anchor=NW, image=img4)
    lb= Label(canvas,text='Horizontal Flip :')
    lb.config(font=("Courier", 15))
    lb.place(x=760,y=20)

    img5= Image.open(filename)
    img5 = img5.resize((300, 300), Image.ANTIALIAS)
    img5= ImageTk.PhotoImage(img5)
    canvas.create_image(760, 400, anchor=NW, image=img5)
    lb= Label(canvas,text='Horizontal flip 2:')
    lb.config(font=("Courier", 15))
    lb.place(x=760,y=380)

    Predict =Button(canvas,text="Predict",anchor=W,command=partial(change2,root2,filename)).place(x=500,y=320)
    

    root2.mainloop()


def browseImage():
    
    filename = filedialog.askopenfilename(initialdir=r"D:\BE_PROJECT\-",title="Select File",filetype=(("jpeg files","*.jpg"),("all files","*.*")))
    panel.destroy()
    newwin(filename)

    
    
    
    

def openCamera():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Leaf Recognition")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
           # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "example{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
    img_name=r"D:\BE_PROJECT\\"+img_name
    newwin(img_name)


    




browse = Button(master,text="Browse",command=browseImage)
browse.place(x=60,y=380)

capture = Button(master,text="capture",command=openCamera)
capture.place(x=260,y=380)

nextb = Button(master,text="next >")
nextb.place(x=180,y=460)

img = ImageTk.PhotoImage(Image.open("noimg.jpg"))
panel = Label(master, image = img)
panel.place(x=40,y=40)

select =Label(master,text="Selected Image",font=("Courier", 15))
select.place(x=120,y=20)

master.mainloop()
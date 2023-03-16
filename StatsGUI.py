import tensorflow as tf
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from PIL import Image
import dill as pickle
import cv2
from sklearn.utils import Bunch
from skimage.io import imread
from skimage.transform import resize
import joblib
from sklearn import preprocessing
import time
import pandas as pd

canvas_width = 550
canvas_height = 300

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
BG_GRAY_DARK = "#A9A9A9"
TEXT_COLOR = "#EAECEE"
PRED_COLOR = "#FFFFFF"
FONT = "Helvetica 8 italic"
FONT_ALGO = "Helvetica 12"
FONT_BOLD = "Helvetica 13 bold"
OPTIONS = ["Digit Recognizer", "Fuit Recognizer"]


class GUI:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.FILENAME = ""
        self.window.title("Bio-Inspired Feature Selection")
        self.window.resizable(width = False, height = False)
        self.window.configure(width = 900, height = 1000, bg = BG_COLOR)

        head_label = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "Cancer Detector", font = FONT_BOLD, pady = 10)
        head_label.place(relwidth = 1)

        line = Label(self.window, width = 450, bg = BG_GRAY)
        line.place(relwidth = 1, rely = 0.07, relheight = 0.012)

        algo_label = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "Enter values in CSV format", font = FONT_ALGO, pady = 5)
        algo_label.place(relx = 0.4, rely = 0.1)

        populate_button = Button(self.window, text = "Populate", width = 20, font = FONT_BOLD, bg = BG_GRAY_DARK, command=self.populate)
        populate_button.place(relx = 0.78, rely=0.1, relwidth = 0.15, relheight = 0.03)

        self.textbox = Text(self.window,height=2.5, width=100)
        self.textbox.place(relx=0.05, rely=0.125)

        label0 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "radius_mean", font = FONT_ALGO, pady = 2)
        label0.place(relx = 0.05, rely = 0.20)
        self.textbox0 = Text(self.window,height=0.5, width=10)
        self.textbox0.place(relx=0.25, rely=0.20)

        label1 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "texture_mean", font = FONT_ALGO, pady = 2)
        label1.place(relx = 0.05, rely = 0.24)
        self.textbox1 = Text(self.window,height=0.5, width=10)
        self.textbox1.place(relx=0.25, rely=0.24)

        label2 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "perimeter_mean", font = FONT_ALGO, pady = 2)
        label2.place(relx = 0.05, rely = 0.28)
        self.textbox2 = Text(self.window,height=0.5, width=10)
        self.textbox2.place(relx=0.25, rely=0.28)
        
        label3 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "area_mean", font = FONT_ALGO, pady = 2)
        label3.place(relx = 0.05, rely = 0.32)
        self.textbox3 = Text(self.window,height=0.5, width=10)
        self.textbox3.place(relx=0.25, rely=0.32)

        label4 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "smoothness_mean", font = FONT_ALGO, pady = 2)
        label4.place(relx = 0.05, rely = 0.36)
        self.textbox4 = Text(self.window,height=0.5, width=10)
        self.textbox4.place(relx=0.25, rely=0.36)

        label5 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "compactness_mean", font = FONT_ALGO, pady = 2)
        label5.place(relx = 0.05, rely = 0.40)
        self.textbox5 = Text(self.window,height=0.5, width=10)
        self.textbox5.place(relx=0.25, rely=0.40)

        label6 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "concavity_mean", font = FONT_ALGO, pady = 2)
        label6.place(relx = 0.05, rely = 0.44)
        self.textbox6 = Text(self.window,height=0.5, width=10)
        self.textbox6.place(relx=0.25, rely=0.44)
        
        label7 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "concave points_mean", font = FONT_ALGO, pady = 2)
        label7.place(relx = 0.05, rely = 0.48)
        self.textbox7 = Text(self.window,height=0.5, width=10)
        self.textbox7.place(relx=0.25, rely=0.48)

        label8 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "symmetry_mean", font = FONT_ALGO, pady = 2)
        label8.place(relx = 0.05, rely = 0.52)
        self.textbox8 = Text(self.window,height=0.5, width=10)
        self.textbox8.place(relx=0.25, rely=0.52)
        
        label9 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "fractal_dimension_mean", font = FONT_ALGO, pady = 2)
        label9.place(relx = 0.05, rely = 0.56)
        self.textbox9 = Text(self.window,height=0.5, width=10)
        self.textbox9.place(relx=0.25, rely=0.56)

        label10 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "radius_se", font = FONT_ALGO, pady = 2)
        label10.place(relx = 0.05, rely = 0.60)
        self.textbox10 = Text(self.window,height=0.5, width=10)
        self.textbox10.place(relx=0.25, rely=0.60)
        
        label11 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "texture_se", font = FONT_ALGO, pady = 2)
        label11.place(relx = 0.05, rely = 0.64)
        self.textbox11 = Text(self.window,height=0.5, width=10)
        self.textbox11.place(relx=0.25, rely=0.64)

        label12 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "perimeter_se", font = FONT_ALGO, pady = 2)
        label12.place(relx = 0.05, rely = 0.68)
        self.textbox12 = Text(self.window,height=0.5, width=10)
        self.textbox12.place(relx=0.25, rely=0.68)

        label13 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "area_se", font = FONT_ALGO, pady = 2)
        label13.place(relx = 0.05, rely = 0.72)
        self.textbox13 = Text(self.window,height=0.5, width=10)
        self.textbox13.place(relx=0.25, rely=0.72)

        label14 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "smoothness_se", font = FONT_ALGO, pady = 2)
        label14.place(relx = 0.55, rely = 0.20)
        self.textbox14 = Text(self.window,height=0.5, width=10)
        self.textbox14.place(relx=0.75, rely=0.20)
        
        label15 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "compactness_se", font = FONT_ALGO, pady = 2)
        label15.place(relx = 0.55, rely = 0.24)
        self.textbox15 = Text(self.window,height=0.5, width=10)
        self.textbox15.place(relx=0.75, rely=0.24)

        label16 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "concavity_se", font = FONT_ALGO, pady = 2)
        label16.place(relx = 0.55, rely = 0.28)
        self.textbox16 = Text(self.window,height=0.5, width=10)
        self.textbox16.place(relx=0.75, rely=0.28)

        label17 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "concave_points_se", font = FONT_ALGO, pady = 2)
        label17.place(relx = 0.55, rely = 0.32)
        self.textbox17 = Text(self.window,height=0.5, width=10)
        self.textbox17.place(relx=0.75, rely=0.32)

        label18 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "symmetry_se", font = FONT_ALGO, pady = 2)
        label18.place(relx = 0.55, rely = 0.36)
        self.textbox18 = Text(self.window,height=0.5, width=10)
        self.textbox18.place(relx=0.75, rely=0.36)

        label19 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "fractal_dimension_se", font = FONT_ALGO, pady = 2)
        label19.place(relx = 0.55, rely = 0.40)
        self.textbox19 = Text(self.window,height=0.5, width=10)
        self.textbox19.place(relx=0.75, rely=0.40)

        label20 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "radius_worst", font = FONT_ALGO, pady = 2)
        label20.place(relx = 0.55, rely = 0.44)
        self.textbox20 = Text(self.window,height=0.5, width=10)
        self.textbox20.place(relx=0.75, rely=0.44)
        
        label21 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "texture_worst", font = FONT_ALGO, pady = 2)
        label21.place(relx = 0.55, rely = 0.48)
        self.textbox21 = Text(self.window,height=0.5, width=10)
        self.textbox21.place(relx=0.75, rely=0.48)

        label22 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "perimeter_worst", font = FONT_ALGO, pady = 2)
        label22.place(relx = 0.55, rely = 0.52)
        self.textbox22 = Text(self.window,height=0.5, width=10)
        self.textbox22.place(relx=0.75, rely=0.52)

        label23 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "area_worst", font = FONT_ALGO, pady = 2)
        label23.place(relx = 0.55, rely = 0.56)
        self.textbox23 = Text(self.window,height=0.5, width=10)
        self.textbox23.place(relx=0.75, rely=0.56)

        label24 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "smoothness_worst", font = FONT_ALGO, pady = 2)
        label24.place(relx = 0.55, rely = 0.60)
        self.textbox24 = Text(self.window,height=0.5, width=10)
        self.textbox24.place(relx=0.75, rely=0.60)

        label25 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "compactness_worst", font = FONT_ALGO, pady = 2)
        label25.place(relx = 0.55, rely = 0.64)
        self.textbox25 = Text(self.window,height=0.5, width=10)
        self.textbox25.place(relx=0.75, rely=0.64)
        
        label26 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "concavity_worst", font = FONT_ALGO, pady = 2)
        label26.place(relx = 0.55, rely = 0.68)
        self.textbox26 = Text(self.window,height=0.5, width=10)
        self.textbox26.place(relx=0.75, rely=0.68)

        label27 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "concave_points_worst", font = FONT_ALGO, pady = 2)
        label27.place(relx = 0.55, rely = 0.72)
        self.textbox27 = Text(self.window,height=0.5, width=10)
        self.textbox27.place(relx=0.75, rely=0.72)

        label28 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "symmetry_worst", font = FONT_ALGO, pady = 2)
        label28.place(relx = 0.55, rely = 0.76)
        self.textbox28 = Text(self.window,height=0.5, width=10)
        self.textbox28.place(relx=0.75, rely=0.76)

        label29 = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "fractal_dimension_worst", font = FONT_ALGO, pady = 2)
        label29.place(relx = 0.05, rely = 0.76)
        self.textbox29 = Text(self.window,height=0.5, width=10)
        self.textbox29.place(relx=0.25, rely=0.76)

        # self.drop_val = StringVar(self.window)
        # self.drop_val.set("Select an option")
        # self.drop = OptionMenu(self.window, self.drop_val, *OPTIONS)
        # self.drop.place(relx = 0.525, rely = 0.125)
        # self.drop.configure(width = 15, bg = BG_COLOR, fg = TEXT_COLOR)

        # self.canvas_widget = Canvas(self.window, width = 20, height = 2)
        # self.canvas_widget.place(relheight = 0.54, relwidth = 1, rely = 0.2)

        bottom_label = Label(self.window, bg = BG_GRAY_DARK, height = 80)
        bottom_label.place(relwidth = 1, rely = 0.825)

        # browse_button = Button(self.window, text = "Browse", width = 20, font = FONT_BOLD, bg = BG_GRAY_DARK, command = self.browseFiles)
        # browse_button.place(relx = 0.385, rely = 0.75, relwidth = 0.25, relheight = 0.04)

        send_button = Button(bottom_label, text = "Predict", width = 20, font = FONT_BOLD, bg = BG_GRAY_DARK, command = self.predict)
        send_button.place(rely = 0.001, relwidth = 1, relheight = 0.03)

        self.msg_entry = Entry(bottom_label, disabledbackground = "#17202A", fg = PRED_COLOR, font = FONT_BOLD)
        self.msg_entry.place(rely = 0.03, relheight = 0.04, relwidth = 1)
        self.msg_entry.configure(state = DISABLED)


    def predict(self):
        values = self.textbox.get("1.0",END).split(',')
        if len(values) != 30:
            tk.messagebox.showerror(title = "No Image", message = "Image not selected")
            return
        selected_features = ['texture_mean', 'perimeter_mean', 'symmetry_mean', 'fractal_dimension_mean', 'perimeter_worst', 'fractal_dimension_worst']
        features = {0: "radius_mean", 1: "texture_mean", 2:	"perimeter_mean", 3: "area_mean", 4: "smoothness_mean", 5: "compactness_mean", 6: "concavity_mean",
            7: "concave points_mean", 8: "symmetry_mean", 9: "fractal_dimension_mean", 10: "radius_se", 11: "texture_se", 12: "perimeter_se", 13: "area_se",
            14: "smoothness_se", 15: "compactness_se", 16: "concavity_se", 17: "concave_points_se", 18: "symmetry_se", 19: "fractal_dimension_se",
            20: "radius_worst", 21: "texture_worst", 22: "perimeter_worst", 23: "area_worst", 24: "smoothness_worst", 25: "compactness_worst",
            26:"concavity_worst", 27: "concave_points_worst", 28: "symmetry_worst", 29: "fractal_dimension_worst"}
        input_values = []
        for index, value in enumerate(values):
            if features[index] in selected_features:
                input_values.append(float(value))
        X = pd.DataFrame(input_values)
        d = preprocessing.normalize(X)
        loaded_model = tf.keras.models.load_model('finalized_model.pkl')
        print(loaded_model.predict(tf.reshape(d,(1,-1))))

    def paint(self, event):
       color = "#476042"
       x1, y1 = (event.x - 1), (event.y - 1)
       x2, y2 = (event.x + 1), (event.y + 1)
       self.canvas_widget.create_oval(x1, y1, x2, y2, fill = color)

    def populate(self):
        values = self.textbox.get("1.0",END).split(',')
        print(values)
        for index, value in enumerate(values):
            self.populate_texbox(index, value)

    def clicked(self):
        pass
    
    def populate_texbox(self, index, value):
        if index == 0:
            self.textbox0.insert(tk.END, str(value))
        if index == 1:
            self.textbox1.insert(tk.END, str(value))
        if index == 2:
            self.textbox2.insert(tk.END, str(value))
        if index == 3:
            self.textbox3.insert(tk.END, str(value))
        if index == 4:
            self.textbox4.insert(tk.END, str(value))
        if index == 5:
            self.textbox5.insert(tk.END, str(value))
        if index == 6:
            self.textbox6.insert(tk.END, str(value))
        if index == 7:
            self.textbox7.insert(tk.END, str(value))
        if index == 8:
            self.textbox8.insert(tk.END, str(value))
        if index == 9:
            self.textbox9.insert(tk.END, str(value))
        if index == 10:
            self.textbox10.insert(tk.END, str(value))
        if index == 11:
            self.textbox11.insert(tk.END, str(value))
        if index == 12:
            self.textbox12.insert(tk.END, str(value))
        if index == 13:
            self.textbox13.insert(tk.END, str(value))
        if index == 14:
            self.textbox14.insert(tk.END, str(value))
        if index == 15:
            self.textbox15.insert(tk.END, str(value))
        if index == 16:
            self.textbox16.insert(tk.END, str(value))
        if index == 17:
            self.textbox17.insert(tk.END, str(value))
        if index == 18:
            self.textbox18.insert(tk.END, str(value))
        if index == 19:
            self.textbox19.insert(tk.END, str(value))
        if index == 20:
            self.textbox20.insert(tk.END, str(value))
        if index == 21:
            self.textbox21.insert(tk.END, str(value))
        if index == 22:
            self.textbox22.insert(tk.END, str(value))
        if index == 23:
            self.textbox23.insert(tk.END, str(value))
        if index == 24:
            self.textbox24.insert(tk.END, str(value))
        if index == 25:
            self.textbox25.insert(tk.END, str(value))
        if index == 26:
            self.textbox26.insert(tk.END, str(value))
        if index == 27:
            self.textbox27.insert(tk.END, str(value))
        if index == 28:
            self.textbox28.insert(tk.END, str(value))
        if index == 29:
            self.textbox29.insert(tk.END, str(value))
        

if __name__ == "__main__":
    app = GUI()
    app.run()

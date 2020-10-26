""" Face alignment Demo with GUI"""
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from os import path
import cv2
from matplotlib import pyplot as plt
from detectors import PyramidBox
from utils import draw_bboxes, draw_pts
from faceAlignment import FaceAlignment


class FA:
    def __init__(self):
        super().__init__()
        self.is_loaded = False
        self.FD = PyramidBox(device='cuda')
        self.FA = FaceAlignment(device='cuda')
        self.is_loaded = True


class Demo(tk.Frame):
    def __init__(self, root):
        super().__init__()
        self.root = root
        # Init Models
        self.fa = FA()
        # Init UI
        self.img_file_path = "main.jpg"
        self.text_x = 100
        self.text_y = 15
        tk.Button(text='Open Image File', width=12, command=self.callback_button1).place(x=0, y=0)
        self.master.title("Face Alignment")
        self.pack(fill=tk.BOTH, expand=1)
        self.img = Image.open(self.img_file_path)
        self.tatras = ImageTk.PhotoImage(self.img)
        self.canvas = tk.Canvas(self, width=self.img.size[0], height=self.img.size[1]+30)
        self.canvas_image = self.canvas.create_image(0, 30, anchor=tk.NW, image=self.tatras)
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.set_text("<-- Open File")

    def set_text(self, txt):
        """Set text with image path"""
        self.canvas.delete('text')
        self.canvas.create_text(self.text_x, self.text_y, anchor=tk.W, text=txt, tags='text')

    def load_img(self):
        """Load image by dialog"""
        self.set_text(self.img_file_path)
        self.img = Image.open(self.img_file_path)
        self.tatras = ImageTk.PhotoImage(self.img)
        self.canvas.itemconfig(self.canvas_image, image=self.tatras)
        self.canvas.config(width=self.img.size[0], height=self.img.size[1]+30)

    def callback_button1(self):
        """Running button"""
        if not self.fa.is_loaded:
            self.set_text('Network is not loaded')
            return
        name = tk.filedialog.askopenfilename()                                      # open dialog
        if path.exists(name):
            self.img_file_path = name
            self.load_img()                                                         # load image
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = self.fa.FD.detect_faces(img, conf_th=0.9, scales=[1])          # face detection
            print("{} faces are detected".format(len(bboxes)))
            img_result = draw_bboxes(img, bboxes)                                   # draw face boxes
            pts = self.fa.FA.face_alignment(img, bboxes)                            # face alignment
            print("Face Alignment is done")
            img_result = draw_pts(img_result, pts)                                  # draw landmarks
            plt.figure(1)                                                           # show result
            plt.axis('off')
            plt.imshow(img_result)
            plt.show()


def main():
    root = tk.Tk()
    Demo(root)
    root.mainloop()


if __name__ == '__main__':
    main()

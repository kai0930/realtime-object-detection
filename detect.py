import tkinter as tk
from tkinter import Label, Button
import cv2
import torch
from PIL import Image, ImageTk

# モデルをCPUに強制
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.to('cpu')

cap = cv2.VideoCapture(0)

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("YOLOv5 Real-time Detection")

        self.label = Label(window)
        self.label.pack()

        self.btn = Button(window, text="Quit", command=self.quit)
        self.btn.pack()

        self.window.after(100, self.update)

    def update(self):
        ret, frame = cap.read()
        if ret:
            # 推論を try-except で囲む（安定性向上）
            try:
                results = model(frame)
                img = results.render()[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)

                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
            except Exception as e:
                print("推論エラー:", e)

        self.window.after(30, self.update)

    def quit(self):
        cap.release()
        self.window.destroy()

root = tk.Tk()
app = App(root)
root.mainloop()

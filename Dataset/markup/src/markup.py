import tkinter as tk
from PIL import Image, ImageTk
import os
import sys

# How much photo:
lotsphoto = 0

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class DrawRect(tk.Tk):
    # tagcheaking
    tagnum = 0
    nametag = str()
    tagarr = []
    #
    Coordinates = []
    #
    i = 1
    imagename = './' + str(sys.argv[1]) + '/' + str(i) + '.JPG'

    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        self.im = Image.open(DrawRect.imagename)
        w, h = self.im.size
        self.canvas = tk.Canvas(self, width=w, height=h, cursor="cross")

        self.canvas.pack(side="bottom", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<Return>", self.ready)
        self.bind('<Escape>', self.esc)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self._draw_image()

    def _draw_image(self):
        self.im = Image.open(DrawRect.imagename)
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # rectangle
        DrawRect.nametag = str(DrawRect.tagnum) + 'rect'
        # print(DrawRect.nametag)
        DrawRect.tagarr.append(DrawRect.nametag)
        # print(DrawRect.tagarr)
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red',
                                                 tag=DrawRect.nametag)

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
        # print(self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')
        # tags
        DrawRect.tagnum += 1

        if ((self.start_x != event.x) & (self.start_y != event.y)):
            # get coordinates
            DrawRect.Coordinates.append([self.start_x, self.start_y, event.x, event.y])

    def esc(self, event):
        # print('ESC')
        # print('Input: ', DrawRect.tagarr, DrawRect.tagnum)
        # tags
        DrawRect.tagnum -= 1
        if len(DrawRect.tagarr) != 0:
            # delete rect
            if len(DrawRect.tagarr) == 1:
                self.canvas.delete(DrawRect.tagarr[0])
                # tag
                del DrawRect.tagarr[0]
                # Coordinates
                del DrawRect.Coordinates[0]
            else:
                self.canvas.delete(DrawRect.tagarr[-1])
                # tag
                del DrawRect.tagarr[-1]
                # Coordinates
                del DrawRect.Coordinates[-1]

        # print('Output: ', DrawRect.tagarr, DrawRect.tagnum)



    def ready(self, event):
        print(str(DrawRect.i) + '.JPG Ready!')
        im = Image.open(DrawRect.imagename)
        j = 0

        # file
        file = folder + 'watched.txt'
        f = open(file, 'a')
        f.write(str(DrawRect.i) + '.JPG ')

        for i in DrawRect.Coordinates:
            w_s = min(i[0], i[2])
            w_e = max(i[0], i[2])
            h_s = min(i[1], i[3])
            h_e = max(i[1], i[3])

            name = folder + str(DrawRect.i) + '_' + str(j) + '.jpg'

            j += 1

            im.crop((w_s, h_s, w_e, h_e)).save(name)

            # file
            f.write(str(DrawRect.i) + '_' + str(j) + '.jpg ')

        f.write('\n')
        f.close()

        # UPDATE for loading new image
        if (DrawRect.i != lotsphoto):
            DrawRect.Coordinates = []
            DrawRect.i += 1
            DrawRect.imagename = './' + str(sys.argv[1]) + '/' + str(DrawRect.i) + '.JPG'
            # destroy window
            self.destroy()
            # Start new
            self.__init__()
        else:
            exit(0)


if __name__ == "__main__":
    # get folder
    folderimage = './' + str(sys.argv[1]) + '/'
    # create folder
    folder = './' + str(sys.argv[2]) + '/'
    createFolder(folder)

    lotsphoto = len([name for name in os.listdir(folderimage)
                     if os.path.isfile(os.path.join(folderimage, name))]) - 1
    print(lotsphoto)

    # run application
    app = DrawRect()
    app.mainloop()
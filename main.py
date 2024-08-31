import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from numba import cuda
import logging

# Настройка логгирования
logging.basicConfig(level=logging.DEBUG)

def color_scheme_function(n, scheme):
    if scheme == 0:
        r = int(255 * (n / 256.0))
        g = int(255 * ((n / 256.0) ** 2))
        b = int(255 * ((n / 256.0) ** 3))
    elif scheme == 1:
        r = int(255 * ((n / 256.0) ** 3))
        g = int(255 * (n / 256.0))
        b = int(255 * ((n / 256.0) ** 2))
    elif scheme == 2:
        r = int(255 * ((n / 256.0) ** 2))
        g = int(255 * ((n / 256.0) ** 3))
        b = int(255 * (n / 256.0))
    elif scheme == 3:
        r = int(255 * (n / 256.0))
        g = int(255 * (n / 128.0) % 256)
        b = int(255 * (n / 64.0) % 256)
    else:
        r = int(255 * (n / 64.0) % 256)
        g = int(255 * (n / 128.0) % 256)
        b = int(255 * (n / 256.0))
    return r, g, b

@cuda.jit(device=True)
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

class MandelbrotApp:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.photo = None

        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0

        self.color_scheme = tk.StringVar(value="Scheme 1")
        self.color_schemes = ["Scheme 1", "Scheme 2", "Scheme 3", "Scheme 4", "Scheme 5"]
        self.dropdown = tk.OptionMenu(master, self.color_scheme, *self.color_schemes, command=self.create_fractal)
        self.dropdown.pack()

        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.move_image)
        self.canvas.bind("<Configure>", self.resize)

        # Привязка событий колесика мыши для Windows и других ОС
        self.canvas.bind_all("<MouseWheel>", self.zoom)  # Windows
        self.canvas.bind_all("<Button-4>", self.zoom)    # Linux
        self.canvas.bind_all("<Button-5>", self.zoom)    # Linux

        self.create_fractal()

    @staticmethod
    @cuda.jit
    def mandelbrot_kernel(image, offset_x, offset_y, scale, scheme):
        def color_scheme_function(n, scheme):
            if scheme == 0:
                r = int(255 * (n / 256.0))
                g = int(255 * ((n / 256.0) ** 2))
                b = int(255 * ((n / 256.0) ** 3))
            elif scheme == 1:
                r = int(255 * ((n / 256.0) ** 3))
                g = int(255 * (n / 256.0))
                b = int(255 * ((n / 256.0) ** 2))
            elif scheme == 2:
                r = int(255 * ((n / 256.0) ** 2))
                g = int(255 * ((n / 256.0) ** 3))
                b = int(255 * (n / 256.0))
            elif scheme == 3:
                r = int(255 * (n / 256.0))
                g = int(255 * (n / 128.0) % 256)
                b = int(255 * (n / 64.0) % 256)
            else:
                r = int(255 * (n / 64.0) % 256)
                g = int(255 * (n / 128.0) % 256)
                b = int(255 * (n / 256.0))
            return r, g, b

        x, y = cuda.grid(2)
        if x < image.shape[1] and y < image.shape[0]:
            c = complex((x + offset_x) / (200.0 * scale) - 2.0,
                        (y + offset_y) / (150.0 * scale) - 1.5)
            z = 0
            for n in range(256):  # Ограничение на количество итераций
                if abs(z) > 2:
                    break
                z = z*z + c
            r, g, b = color_scheme_function(n, scheme)
            image[y, x, 0] = r
            image[y, x, 1] = g
            image[y, x, 2] = b

    def create_fractal(self, *args):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(image_array.shape[1] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(image_array.shape[0] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        scheme = self.color_schemes.index(self.color_scheme.get())
        MandelbrotApp.mandelbrot_kernel[blockspergrid, threadsperblock](image_array, self.offset_x, self.offset_y, self.scale, scheme)
        self.image = Image.fromarray(image_array)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def move_image(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.offset_x -= dx  # Инверсия движения
        self.offset_y -= dy  # Инверсия движения
        self.start_x = event.x
        self.start_y = event.y
        self.create_fractal()

    def zoom(self, event):
        logging.debug(f"Mouse wheel event: delta={event.delta}")
        mouse_x, mouse_y = event.x, event.y
        if event.num == 4 or event.delta > 0:
            scale_factor = 1.1
        elif event.num == 5 or event.delta < 0:
            scale_factor = 1 / 1.1
        else:
            return

        # Масштабирование относительно точки под курсором
        self.offset_x = (self.offset_x + mouse_x) * scale_factor - mouse_x
        self.offset_y = (self.offset_y + mouse_y) * scale_factor - mouse_y
        self.scale *= scale_factor
        self.create_fractal()

    def resize(self, event):
        self.create_fractal()

root = tk.Tk()
app = MandelbrotApp(root)
root.mainloop()
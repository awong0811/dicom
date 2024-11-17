import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

def read_dicom(path: str):
    ds = pydicom.dcmread(path)
    print(type(ds.PixelData))
    print(len(ds.PixelData))
    print(ds.PixelData[:2])
    return ds

def animate(frames, interval=33):
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def animate(i):
        im.set_array(frames[i])
        return im,
    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames), interval=interval)
    return anim

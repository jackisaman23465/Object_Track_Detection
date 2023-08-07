import imageio
import os

images = []

allfiles = os.listdir("sample1")
imlist = [filename for filename in allfiles if filename[-4:] in [".jpg"]]

for filename in imlist:
    images.append(imageio.imread("sample1/"+filename))
imageio.mimsave('movie.gif', images)
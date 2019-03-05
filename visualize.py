import sys
from scipy import misc
from matplotlib import pyplot as plt
import cv2

gt = {}
junk = {}

f = open("hard_ok.txt", "r")
lines = [line[:-1] for line in f.readlines()]
for line in lines:
    parts = line.split(",")
    q = parts[0]
    candidates = parts[1].strip().split()
    gt[q] = candidates
f.close()


f = open("hard_junk.txt", "r")
lines = [line[:-1] for line in f.readlines()]
for line in lines:
    parts = line.split(",")
    q = parts[0]
    candidates = parts[1].strip().split()
    junk[q] = candidates


f = open(sys.argv[1], "r")
lines = [line[:-1] for line in f.readlines()]
lines = lines[1:]
prefix = "/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/landmarks/oxfordRe/jpg/"
for line in lines:
    parts = line.split(",")
    q = parts[0]
    #if q != "balliol_000194" and q != "ashmolean_000269" and q != "radcliffe_camera_000286":
    #    continue
    #if q != "ashmolean_000269":
    #    continue
    #if q != "radcliffe_camera_000286":
    #    continue
    #if q != "balliol_000187":
    #    continue
    candidates = parts[1].strip().split()
    
    #visualize code here!
    
    ###########################################  plotting config
    plotted = 0      # plot query!!!!
    fig=plt.figure(q, figsize=(16, 10))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4)
    columns = 10
    rows = 5
    top, bottom, left, right = [10]*4
    num_images_to_show = columns * rows
    image_size = 300
    ###########################################

    
    img = misc.imread(prefix + q + ".jpg")
    img = misc.imresize(img, (image_size,image_size,3))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 255]) #blue
    ax = fig.add_subplot(rows, columns, plotted + 1)
    #ax.set_title(q)
    plt.axis('off')
    plt.imshow(img)
    plotted += 1
    for c in candidates:
        if c in junk or c in junk[q]:
            continue
        try:
            img = misc.imread(prefix + c + ".jpg")
        except:
            continue
        img = misc.imresize(img, (image_size,image_size,3))
        if c in gt[q]:
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]
        # border widths; I set them all to 150
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        ax = fig.add_subplot(rows, columns, plotted + 1) 
        #ax.set_title(c)
        plt.axis('off')
        plt.imshow(img)
        plotted += 1
        if plotted == num_images_to_show:
            break
    plt.show()
        
        
        

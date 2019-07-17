########################################################################################################################
# GRAIN EVALUATOR
########################################################################################################################
# Author: Christian Hofmann
# Date: 06/2019
# Dependencies: Python3, openCV (cv2), matplotlib, numpy, pandas
#
###########################################################
#  1  IMPORT DEPENDENCIES
###########################################################
import numpy as np
import cv2
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.widgets import Slider
from scipy.stats import kde
import tkinter as tk
from tkinter import filedialog

###########################################################
#  2  INPUT FILES
###########################################################

# samples_dir = 'samples/'
# master = '13.tif'
# files = ['13_Si.tif']
# files = ['11_Al.tif', '11_C.tif', '11_Ca.tif', '11_O.tif', '11_Si.tif']
# master = samples_dir + '13.tif'
# files = samples_dir + ['13_Si.tif']

###########################################################
#  3  PARAMETER SETUP
###########################################################
#th = 100  # Default Threshold Value
#  CLAHE PARAMETERS (Histogram Equilization)
CLAHE = True  # True/False (On/Off)
cliplimit = 1  # CLAHE Clip limit
tileGridSize = 8  # CLAHE kernel size
#  GAUSS-BLUR PARAMETERS
kernelsize = 3
#  HISTOGRAM PARAMETERS
binwidth = 2


###########################################################
#  4  VARIABLES & CLASS & FUNCTIONS
###########################################################
# master = samples_dir + master
# files = samples_dir + files
# files = [samples_dir + x for x in files]

# CONTOUR modes:
# cv2.CHAIN_APPROX_SIMPLE
# cv2.CHAIN_APPROX_NONE
def contours(aimg, min_area = 1, mode=cv2.CHAIN_APPROX_SIMPLE):
    contours, hierarchy = cv2.findContours(aimg, cv2.RETR_EXTERNAL, mode)
    cnts_filter = []
    for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cnts_filter.append(cnt)
    return cnts_filter

def evaluate(cnts, mpp):
    data = pandas.DataFrame(columns=['cx', 'cy', 'contour area [px²]', 'contour area [µm²]', 'fitting area [px²]',
                                     'fitting area [µm²]', 'size by fitting [px]', 'size by fitting [µm]',
                                     'size by area [px]', 'size by area [µm]', 'fitting orientation [°]'])
    for i, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        # print(i,area)
        # CONTOUR:
        (cenx, ceny) = centroid(cnt)
        # FITTINGS:
        if len(cnt) < min_ellipse_points:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            r = radius  # * mpp #micrometers per pixel
            fitarea = np.pi * r ** 2
            a = 'circle'
        else:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (w, h), a = ellipse
            a = round(a, 3)
            radius = min(w, h) / 2
            r = radius  # * mpp #micrometers per pixel
            fitarea = np.pi * w / 2 * h / 2
            if fitarea == 0:
                fitarea = 1
        # SIZES:
        size_fit = round(r * 2, 2)
        size_area = round(np.sqrt(area / np.pi) * 2, 2)
        # OTHER:
        area_fit_quot = round(area / fitarea, 3)
        data.loc[i] = [cenx, ceny, area, area * mpp * mpp, fitarea, fitarea * mpp * mpp, size_fit, size_fit * mpp,
                       size_area, size_area * mpp, a]
    return data



def maxpixel(aimg):
    maximum = 0
    for a in aimg:
        ma = max(a)
        if ma > maximum:
            maximum = ma
    return maximum

# THRESHOLD modes:
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV
def thresh(aimg, value=50, type=cv2.THRESH_TOZERO):
    maximum = maxpixel(aimg)
    threshold = int(value / 100 * maximum)
    ret, timg = cv2.threshold(aimg, threshold, 255, type)
    return timg


# BLUR (GAUSS)
def blur(aimg, ksize=5):
    blurred = cv2.GaussianBlur(aimg, (ksize, ksize), 0)
    return blurred

# EQUILIZE
def equilize(aimg):
    return cv2.equalizeHist(aimg)


# APPROXIMATION
def approx(contours, epsilon_percent=0.005):
    approx = []
    for cnt in contours:
        epsilon = epsilon_percent * cv2.arcLength(cnt, True)
        approx.append(cv2.approxPolyDP(cnt, epsilon, True))
    return approx


# CENTROID
# Returns the centroid of a contour (openCV-Contour)
def centroid(contour):
    M = cv2.moments(contour)
    cenx = int(M["m10"] / M["m00"])
    ceny = int(M["m01"] / M["m00"])
    return (cenx, ceny)


# DISTANCE between two points (pythagoras)
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# SIZE FROM FITTING
min_ellipse_points = 8
def fit_size(cnt):
    if len(cnt) < min_ellipse_points:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        size = radius * 2
    else:
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (w, h), a = ellipse
        size = min((w, h))
    return size


# DRAW the elements into an image:
# elemnts: ('contour', col), ('fit',(255,255,255)), ('index')
def draw(aimg, all_contours, elements, indexes=[-1]):
    image = aimg.copy()
    for k, e in enumerate(elements):
        for i, cnt in enumerate(all_contours):
            if cv2.contourArea(cnt) >= 1:
                if (i in indexes) or indexes[0] == -1:
                    if e[0] == 'contour':
                        col = e[1]
                        cv2.drawContours(image, [cnt], -1, col, 1)
                    elif e[0] == 'contour_fill':
                        col = e[1]
                        cv2.drawContours(image, [cnt], -1, col, thickness=cv2.FILLED)
                    elif e[0] == 'fit':
                        col = e[1]
                        if len(cnt) < min_ellipse_points:
                            (x, y), radius = cv2.minEnclosingCircle(cnt)
                            cv2.circle(image, (int(x), int(y)), int(radius), col, 1)
                        else:
                            ellipse = cv2.fitEllipse(cnt)
                            cv2.ellipse(image, ellipse, col, 1)
                    elif e[0] == 'index':
                        col = e[1]
                        (cenx, ceny) = centroid(cnt)
                        cv2.putText(image, '{}'.format(i), (int(cenx), int(ceny)), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                                    col)
                    elif e[0] == 'angle':
                        if len(cnt) > min_ellipse_points:
                            col = e[1]
                            ellipse = cv2.fitEllipse(cnt)
                            (x1, y1), (w, h), a = ellipse
                            x2 = x1 + max(w,h)/2*np.sin(a*np.pi/180)
                            y2 = y1 - max(w,h)/2*np.cos(a*np.pi/180)
                            cv2.line(image,(int(x1),int(y1)),(int(x2),int(y2)),col,1)

    return image


# LOAD image
GRAYSCALE = cv2.IMREAD_GRAYSCALE
COLOR = cv2.IMREAD_COLOR


def load(filename, mode=COLOR):
    img = cv2.imread(filename, mode)  # returns BGR Image
    if mode == COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# CONVERT COLOR to GRAY
def gray(aimg):
    return cv2.cvtColor(aimg, cv2.COLOR_RGB2GRAY)


# CONVERT GRAY TO RGB
def rgb(aimg):
    return cv2.cvtColor(aimg, cv2.COLOR_GRAY2RGB)


# HEIGHT and WIDTH of image
def img_size(aimg):
    if len(aimg.shape) == 2:
        h, w = aimg.shape
    elif len(aimg.shape) == 3:
        h, w, ch = aimg.shape
    return (h, w)


# CROP PIXELS from BOTTOM, TOP, LEFT or RIGHT
def crop(aimg, b=0, t=0, l=0, r=0):
    img = aimg.copy()
    (h, w) = img_size(img)
    img = img[t:h - b, l:w - r]
    return img


def show(aimg, Maximize=True, Show=True,AxisVisible=True):
    fig, axarr, ims = shown([aimg], Maximize, Show,AxisVisible=AxisVisible)
    return fig, axarr, ims


def shown(aimgs, Show=True, Maximize=True, c=0, r=0, AxisVisible=True, Title='Image'):
    n = len(aimgs)
    if (c == 0) and (r == 0):
        if n == 1:
            c = 1
            r = 1
        elif n == 2:
            c = 2
            r = 1
        elif n == 3:
            c = 3
            r = 1
        elif n == 4:
            c = 2
            r = 2
        elif n == 5:
            c = 3
            r = 2
        elif n == 6:
            c = 3
            r = 2
        elif n == 7:
            c = 4
            r = 2
        elif n == 8:
            c = 4
            r = 2
        elif n == 9:
            c = 3
            r = 3
    fig, axarr = plt.subplots(r, c, sharex=True, sharey=True)
    fig.canvas.set_window_title(Title)

    if n == 1:
        axarr = [axarr]
    else:
        axarr = axarr.flatten()
    ims = []
    for ii, img in enumerate(aimgs):
        axarr[ii].xaxis.set_visible(AxisVisible)
        axarr[ii].yaxis.set_visible(AxisVisible)
        if len(img.shape) == 2:
            ims.append(axarr[ii].imshow(img,cmap=plt.cm.gray))
        else:
            ims.append(axarr[ii].imshow(img))
    fig.tight_layout()
    if Maximize:
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
    if Show:
        plt.show()
    return fig, axarr, ims

def showmaximize():
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.show()


def split(contours, ki=0.2, ks=0.1):
    # print("L: {}".format(len(contour)))
    contours_new = []
    for n, contour in enumerate(contours):
        if fit_size(contour) < 10:
            contours_new.append(contour)
            continue
        # print(contour.shape)
        dists = []
        count = len(contour)
        for i in range(0, count - 1, 1):
            for j in range(i + 1, count - 1, 1):
                idist = j - i  # "index-distanz"
                if idist > ki * count:
                    p1 = contour[i][0]
                    p2 = contour[j][0]
                    d = dist(p1, p2)  # calculates the distance between two points using simple pythagoras
                    if d <= fit_size(contour) * ks:
                        dists.append([d, i, j])
                        # print('{}-{}: {}'.format(i,j,d))
        # print(dists)
        if len(dists) > 0:
            dists = sorted(dists, key=lambda a: a[0])
            a = dists[0][1]
            b = dists[0][2]
            old = contour.copy()
            new = old[a:b]
            old1 = old[0:a]
            old2 = old[b:len(old)]
            old = np.concatenate((old1, old2))
            contours_new.append(old)
            contours_new.append(new)
        else:
            contours_new.append(contour)
    # print(contours_new)
    # for d in dists:
    #    print('{}, {}-{}')
    # print('OLD: {} - NEW: {}'.format(len(contours),len(contours_new)))
    print('Splitted {} contours to {}'.format(len(contours), len(contours_new)))
    return contours_new
    # print(dists)


# OUT OF ORDER!!!!
# listcentuples = List of tuples (cenx, ceny) centroid coordinates
def Density(aimg, x, y):
    img = aimg.copy()
    nbins = 128
    # x = [cen[0] for cen in listcentuples]
    # y = [cen[1] for cen in listcentuples]
    k = kde.gaussian_kde([x, y])
    Z = np.reshape(k([x, y].T, ))
    print(k)
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # print(xi)
    for i in xi:
        print(i)
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi / max(zi) * 255
    print(zi)
    print(max(zi))
    i = 0
    for ax in xi:
        for ay in yi:
            img[int(ay), int(ax)] = int(zi[i])
            print('{}, {}, {}'.format(ax, ay, zi))
            i += 1
    plt.imshow(img)
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.show()


###########################################################
#  5  PROCESSING & PLOTTING
###########################################################
if __name__ == '__main__':
    # initial GUI:
    window = tk.Tk()
    window.title ('EICHKÖRNCHEN - Grain analysis tool')
    
    # MASTER IMAGE ENTRY
    lbl = tk.Label(window,text='EICHKÖRNCHEN - Grain analysis tool', justify=tk.LEFT, font='Arial 12 bold')
    lbl.grid(column=0, row=0, sticky=tk.W)
    lbl = tk.Label(window,text='Original Image:', justify=tk.LEFT, font='Arial 12 bold')
    lbl.grid(column=0, row=1, sticky=tk.W)
    lbl = tk.Label(window, text = 'The original image is only used for visualizing and is not used for evaluation.', justify=tk.LEFT)
    lbl.grid(column=0, row=2, sticky=tk.W)
    txt = tk.Entry(window,width=30)
    txt.grid(column=0, row=3,sticky=tk.N+tk.S+tk.W+tk.E)
    btn = tk.Button(window, text='Browse...')
    btn.grid(column=1, row=3)
    
    # ISOLATED IMAGE ENTRY
    lbl = tk.Label(window, text='Isolated Image:', justify=tk.LEFT, font='Arial 12 bold')
    lbl.grid(column=0, row=4, sticky=tk.W)
    lbl = tk.Label(window, text='This image should only contain a single channel with the color of the grains to be analized.',
                   justify=tk.LEFT)
    lbl.grid(column=0, row=5, sticky=tk.W)
    txt = tk.Entry(window,width=30)
    txt.grid(column=0, row=6,sticky=tk.N+tk.S+tk.W+tk.E)
    btn = tk.Button(window, text='Browse...')
    btn.grid(column=1, row=6)
    
    # EXPORTS
    lbl = tk.Label(window, text='Exports:', justify=tk.LEFT, font='Arial 12 bold')
    lbl.grid(column=0, row=7, sticky=tk.W)
    chb = tk.Checkbox(window, text='Excel grain table')
    chb.grid(column=0, row=8, sticky=tk.W)
    txt = tk.Entry(window)
    txt.grid(column=1, row=8,sticky=tk.N+tk.S+tk.W+tk.E)
    btn = tk.Button(window, text='Browse...')
    btn.grid(column=2, row=8)
    
    # HINTS
    lbl = tk.Label(window, text='Hints:', justify=tk.LEFT, font='Arial 12 bold')
    lbl.grid(column=0, row=9, sticky=tk.W)
    lbl = tk.Label(window, text='Lorem Ispum',
                   justify=tk.LEFT)
    lbl.grid(column=0, row=10, sticky=tk.W)
    window.mainloop()






    # input files
    root = tk.Tk()
    root.withdraw()
    print('SELECT MASTER IMAGE FILE (RGB COLOR IMAGE):')
    file_master = ''
    while file_master == '':
        file_master = filedialog.askopenfilename(title='SELECT MASTER FILE (RGB COLOR IMAGE)')
    print(file_master)
    print('SELECT GRAYSCALES (COLOR FILTERED IMAGE):')
    files = []
    while len(files) < 1:
        files = filedialog.askopenfilenames(title='SELECT GRAYSCALES (COLOR FILTERED)')
    files = files[::-1]
    print(files)
    immaster = load(file_master, COLOR)
    root.destroy()
    # cropping with zoom rect (get_ylim, get_xlim)
    fig, axarr, ims = show(immaster,AxisVisible=False)
    dx = axarr[0].get_xlim()
    x1 = int(np.ceil(dx[0]))
    x2 = int(np.ceil(dx[1]))
    dy = axarr[0].get_ylim()
    y1 = int(np.ceil(dy[0]))
    y2 = int(np.ceil(dy[1]))
    print((x1, x2), (y1, y2))
    (h, w) = img_size(immaster)
    l = x1
    r = w - x2
    t = y2
    b = h - y1
    immaster = crop(immaster, b, t, l, r)
    #mpp = 200 / 282  # calibration (micrometers per pixel)
    while True:
        try:
            mpp = float(input('MICROMETERS PER PIXEL (µm/pixel): '))
            break
        except ValueError:
            'ENTER FLOAT WITH POINT DECIMAL PLEASE!'
        except TypeError:
            'ENTER FLOAT WITH POINT DECIMAL PLEASE!'
    for f in files:
        # ############
        # THRESHOLDING
        # ############
        imrgb = load(f, COLOR)
        imrgb = crop(imrgb, b, t, l, r)
        imgray = gray(imrgb)
        # DETECTING CONTOURS
        threshold = 50
        blurk = 5
        imthresh = thresh(blur(imgray,blurk), threshold)
        imcanvas = rgb(imthresh)
        cnts = approx(contours(imthresh,cv2.CHAIN_APPROX_NONE),0.005)
        imcanvas = draw(imcanvas,cnts,[('contour',(0,255,255)),('fit',(0,255,0))]) #rechts
        imcanvas2 = draw(immaster, cnts, [('contour', (255, 255, 255))]) #links
        fig, axarr, ims = shown([imcanvas2,imcanvas],False,AxisVisible=False)
        # Slider THRESHOLD
        ax_slid_th = plt.axes([0.1, 0.01, 0.8, 0.02])  # left, bottom, width, height
        slid_th = Slider(ax_slid_th, 'Threshold', 0, 100, valinit=50)
        def slid_th_update(value):
            global threshold
            threshold = value
            imthresh = thresh(blur(imgray,blurk),value)
            imcanvas = rgb(imthresh)
            cnts = contours(imthresh,cv2.CHAIN_APPROX_NONE)
            imcanvas = draw(imcanvas,cnts,[('contour',(0,255,255)),('fit',(0,255,0))]) #rechts
            imcanvas2 = draw(immaster,cnts,[('contour',(255,255,255))]) #links
            ims[0].set_data(imcanvas2)
            ims[1].set_data(imcanvas)
        slid_th.on_changed(slid_th_update)
        # Slider BLUR KERNEL
        ax_slid_blurk = plt.axes([0.1, 0.03, 0.8, 0.02])  # left, bottom, width, height
        height, width = img_size(immaster)
        slid_blurk = Slider(ax_slid_blurk, 'Blur kernel size', 1, int(max([height, width])/10), valinit=5)
        def slid_blurk_update(value):
            global blurk
            value = int(value)
            if value % 2 == 0:
                if value - 1 < 1:
                    value = value + 1
                else:
                    value = value - 1
            #slid_blurk.set
            blurk = value
            imthresh = thresh(blur(imgray,value),threshold)
            imcanvas = rgb(imthresh)
            cnts = contours(imthresh,cv2.CHAIN_APPROX_NONE)
            imcanvas = draw(imcanvas,cnts,[('contour',(0,255,255)),('fit',(0,255,0))]) #rechts
            imcanvas2 = draw(immaster,cnts,[('contour',(255,255,255))]) #links
            ims[0].set_data(imcanvas2)
            ims[1].set_data(imcanvas)
        slid_blurk.on_changed(slid_blurk_update)
        # SHOW THRESHOLDING PLOT
        plt.show()
        # ############
        # EVALUATION
        # ############
        print('Evaluating...')
        data = pandas.DataFrame(columns=['cx', 'cy', 'contour area [px²]', 'contour area [µm²]', 'fitting area [px²]',
                                         'fitting area [µm²]', 'size by fitting [px]', 'size by fitting [µm]',
                                         'size by area [px]', 'size by area [µm]', 'fitting orientation [°]'])
        cnts_filter = []
        for cnt in cnts:
            if cv2.contourArea(cnt) >= 1:
                cnts_filter.append(cnt)
        for i, cnt in enumerate(cnts_filter):
            area = cv2.contourArea(cnt)
            # print(i,area)
            # CONTOUR:
            (cenx, ceny) = centroid(cnt)
            # FITTINGS:
            if len(cnt) < min_ellipse_points:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                r = radius # * mpp #micrometers per pixel
                fitarea = np.pi * r ** 2
                a = 'circle'
            else:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (w, h), a = ellipse
                a = round(a,3)
                radius = min(w, h) / 2
                r = radius #* mpp #micrometers per pixel
                fitarea = np.pi * w / 2 * h / 2
                if fitarea == 0:
                    fitarea = 1
            # SIZES:
            size_fit = round(r * 2, 2)
            size_area = round(np.sqrt(area / np.pi) * 2, 2)
            # OTHER:
            area_fit_quot = round(area / fitarea, 3)
            data.loc[i] = [cenx, ceny, area, area*mpp*mpp, fitarea, fitarea*mpp*mpp, size_fit, size_fit * mpp, size_area, size_area*mpp, a]
        # ############
        # EXPORT
        # ############
        imcanvas = draw(immaster,cnts_filter,(('contour',(255,255,255)),('fit',(0,255,0)),('index',(255,255,255))))
        show(imcanvas)
        try:
            data.to_excel(f + '.xlsx')
            print('Excel-Export: {}'.format(f + '.xlsx'))
        except:
            print('Excel-Export failed!')
        try:
            cv2.imwrite(f + '-eval.tiff', imcanvas)
            print('Evaluate-Image-Export: {}'.format(f + '-eval.tiff'))
        except:
            print('Evaluate-Image-Export failed!')







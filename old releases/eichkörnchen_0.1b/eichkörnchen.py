import gui  # PyQt-GUI: gui.py
import pycv
import matplotlib.pyplot as plt
import cv2
import numpy
import os
from matplotlib.widgets import Slider, CheckButtons, Button
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets


class Eichkoernchen(gui.Ui_MainWindow):
    def __init__(self):
        super(gui.Ui_MainWindow, self).__init__()
        self.imoriginal = None
        self.imisolated = None
        self.mpp = 1

    def setupSignals(self):
        # Image input actions
        self.btnImgOriginal.clicked.connect(self.imoriginal_browse)
        self.btnImgIsolated.clicked.connect(self.imisolated_browse)
        self.btnLoadOriginal.clicked.connect(self.imoriginal_load)
        self.btnLoadIsolated.clicked.connect(self.imisolated_load)
        self.btnShowOriginal.clicked.connect(self.imoriginal_show)
        self.btnShowIsolated.clicked.connect(self.imisolated_show)
        # Crop
        self.btnCrop.clicked.connect(self.imisolated_crop)
        # Manipulate
        self.btnManipulate.clicked.connect(self.imisolated_manipulate)
        # Calibration
        self.btnEqual.clicked.connect(self.cal_equal)
        self.btnDetermine.clicked.connect(self.cal_determine)
        # Exports
        self.chbExcel.clicked.connect(self.exp_excel_check)
        self.btnExcel.clicked.connect(self.exp_excel_browse)
        # Go
        self.btnThreshold.clicked.connect(self.go_thresholding)
        self.btnResult.clicked.connect(self.go_result)
        self.btnExport.clicked.connect(self.go_export)

    # 1) Original Image Actions
    def imoriginal_browse(self):
        file, _ = QFileDialog.getOpenFileName(MainWindow, "Open Original Image", "",
                                              "Image File (*.png; *.bmp; *.tif; *.tiff; *.jpeg; *.jpg; *.gif);;All Files (*)")
        self.txtImgOriginal.setText(file)
        self.imoriginal_load()

    def imoriginal_load(self):
        try:
            filename = self.txtImgOriginal.text()
            self.imoriginal = pycv.load(filename)
            self.statusBar.showMessage('Original Image loaded!')
        except:
            self.statusBar.showMessage('Error while loading Original Image!')

    def imoriginal_show(self):
        if self.imoriginal is not None:
            pycv.shown([self.imoriginal], True, False, AxisVisible=False, Title='Original Image')

    # 2) Isolated Image Actions
    def imisolated_browse(self):
        file, _ = QFileDialog.getOpenFileName(MainWindow, "Open Isolated Image", "",
                                              "Image File (*.png; *.bmp; *.tif; *.tiff; *.jpeg; *.jpg; *.gif);;All Files (*)")
        self.txtImgIsolated.setText(file)
        self.imisolated_load()

    def imisolated_load(self):
        try:
            if self.radImgIsolated1.isChecked():
                filename = self.txtImgOriginal.text()
            elif self.radImgIsolated2.isChecked():
                filename = self.txtImgIsolated.text()
            self.imisolated = pycv.load(filename)
            self.statusBar.showMessage('Isolated Image loaded!')
            # self.txtTop = 0
            # self.txtLeft = 0
            # self.txtBottom = 0
            # self.txtRight = 0
        except:
            self.statusBar.showMessage('Error while loading Isolated Image!')

    def imisolated_show(self):
        if self.imisolated is not None:
            pycv.shown([self.imisolated], True, False, AxisVisible=False, Title='Isolated Image')

    # 3) Cropping Analysis/Isolated Image
    def imisolated_crop(self):
        fig, self.cropax, ims = pycv.shown([self.imisolated], False, True, 0, 0, True, 'CROP ISOLATED IMAGE')
        (h, w) = pycv.img_size(self.imisolated)
        self.cropax[0].set_xlim([int(self.txtLeft.text()), w - int(self.txtRight.text())])
        self.cropax[0].set_ylim([h - int(self.txtBottom.text()), int(self.txtTop.text())])
        ax_apply = plt.axes([0.85, 0.05, 0.1, 0.05])  # left, bottom, width, height
        btnApply = Button(ax_apply, 'APPLY')
        btnApply.on_clicked(self.imisolated_crop_apply)
        ax_reset = plt.axes([0.75, 0.05, 0.1, 0.05])  # left, bottom, width, height
        btnReset = Button(ax_reset, 'RESET')
        btnReset.on_clicked(self.imisolated_crop_reset)
        plt.show()

    def imisolated_crop_apply(self, event):
        dx = self.cropax[0].get_xlim()
        x1 = int(numpy.ceil(dx[0]))
        x2 = int(numpy.ceil(dx[1]))
        dy = self.cropax[0].get_ylim()
        y1 = int(numpy.ceil(dy[0]))
        y2 = int(numpy.ceil(dy[1]))
        # print((x1, x2), (y1, y2))
        (h, w) = pycv.img_size(self.imisolated)
        l = x1
        r = w - x2
        t = y2
        b = h - y1
        if l < 0:
            l = 0
        if r < 0:
            r = 0
        if t < 0:
            t = 0
        if b < 0:
            b = 0
        self.txtLeft.setText(str(l))
        self.txtRight.setText(str(r))
        self.txtTop.setText(str(t))
        self.txtBottom.setText(str(b))
        # immaster = crop(immaster, b, t, l, r)
        plt.close()

    def imisolated_crop_reset(self, event):
        (h, w) = pycv.img_size(self.imisolated)
        self.cropax[0].set_xlim([0, w - 0])
        self.cropax[0].set_ylim([h - 0, 0])

    # 4) Manipulating Isolated Image Channels
    def imisolated_manipulate(self):
        try:
            imr, img, imb = cv2.split(self.imisolated)
            fig, axarr, self.ims = pycv.shown([self.imisolated, imr, img, imb], False, True, AxisVisible=False,
                                              Title='Manipulate Isolated Image')
            ax_s_r = plt.axes([0.1, 0.05, 0.8, 0.02])  # left, bottom, width, height
            ax_s_g = plt.axes([0.1, 0.03, 0.8, 0.02])
            ax_s_b = plt.axes([0.1, 0.01, 0.8, 0.02])
            self.slid_r = Slider(ax_s_r, 'RED', 0.0, 3.0, valinit=1)
            self.slid_g = Slider(ax_s_g, 'GREEN', 0.0, 3.0, valinit=1)
            self.slid_b = Slider(ax_s_b, 'BLUE', 0.0, 3.0, valinit=1)
            self.slid_r.on_changed(self.imisolated_manipulate_update)
            self.slid_g.on_changed(self.imisolated_manipulate_update)
            self.slid_b.on_changed(self.imisolated_manipulate_update)
            fig.canvas.mpl_connect('close_event', self.imisolated_manipulate_close)
            plt.show()
        except:
            print('Error while trying to open manipulate window.')

    def imisolated_manipulate_update(self, value):
        r = self.slid_r.val
        g = self.slid_g.val
        b = self.slid_b.val
        imr, img, imb = cv2.split(self.imisolated)
        imr = imr * r
        imr = imr.astype(numpy.uint8)
        img = img * g
        img = img.astype(numpy.uint8)
        imb = imb * b
        imb = imb.astype(numpy.uint8)
        self.ims[1].set_data(imr)
        self.ims[2].set_data(img)
        self.ims[3].set_data(imb)
        self.im = cv2.merge((imr, img, imb))
        self.ims[0].set_data(self.im)

    def imisolated_manipulate_close(self, evt):
        try:
            self.imisolated = self.im
        except:
            None
        # pycv.shown([self.imisolated], True, True, AxisVisible=False, Title='Isolated Image')

    # 5) Calibration
    def cal_equal(self):
        m = float(self.txtMicrometers.text())
        p = int(self.txtPixels.text())
        self.mpp = m / p
        self.txtMPP.setText(str(self.mpp))

    def cal_determine(self):
        try:
            img = pycv.load(self.txtImgOriginal.text())
            fig, axarr, ims = pycv.shown([img], False, True, AxisVisible=True,
                                         Title='MAKE ZOOM RECTANGLE ON SCALE BAR AND CLOSE')
            plt.title("Zoom to scale bar and click APPLY. Close the window to abort.")
            ax_apply = plt.axes([0.85, 0.05, 0.1, 0.05])  # left, bottom, width, height
            btnApply = Button(ax_apply, 'APPLY')
            btnApply.on_clicked(self.cal_determine_apply)
            plt.show()
            if self.determine_apply:
                dx = axarr[0].get_xlim()
                x1 = int(numpy.ceil(dx[0]))
                x2 = int(numpy.ceil(dx[1]))
                p = x2 - x1
                self.txtPixels.setText(str(p))
                self.btnEqual.click()
        except Exception as e:
            print(e)
            self.statusBar.showMessage('Error while loading Master image for pixel calibration!')

    def cal_determine_apply(self, event):
        self.determine_apply = True
        plt.close()

    # 6) Export Settings
    def exp_excel_check(self):
        if self.chbExcel.isChecked():
            self.txtExcel.setEnabled(True)
            self.btnExcel.setEnabled(True)
        else:
            self.txtExcel.setEnabled(False)
            self.btnExcel.setEnabled(False)

    def exp_excel_browse(self):
        file, _ = QFileDialog.getSaveFileName(MainWindow, "Save Excel to...", "",
                                              "Ecxel-File (*.xlsx);;All Files (*)")
        self.txtExcel.setText(file)

    # 7) Go
    def go_thresholding(self):
        self.im1 = self.imoriginal.copy()
        self.im2 = self.imisolated.copy()

        b = int(self.txtBottom.text())
        t = int(self.txtTop.text())
        l = int(self.txtLeft.text())
        r = int(self.txtRight.text())
        self.im1 = pycv.crop(self.im1, b, t, l, r)
        self.im2 = pycv.crop(self.im2, b, t, l, r)
        self.im2 = pycv.gray(self.im2)  # for analysis
        self.im1_canvas = self.im1.copy()  # original for visual
        self.im2_canvas = pycv.rgb(self.im2.copy())  # analysis for visual
        fig, axarr, self.thresh_ims = pycv.shown([self.im1_canvas, self.im2_canvas], False, True, AxisVisible=False,
                                                 Title='Thresholding')
        # Sliders
        # Slider Threshold
        ax_slid_th = plt.axes([0.1, 0.01, 0.8, 0.02])  # left, bottom, width, height
        self.slid_th = Slider(ax_slid_th, 'THRESHOLD', 0, 100, valinit=50)
        self.slid_th.on_changed(self.go_thresholding_update)
        # Slider GAUSS-BLUR
        ax_slid_blurk = plt.axes([0.1, 0.03, 0.8, 0.02])
        height, width = pycv.img_size(self.im2)
        self.slid_blurk = Slider(ax_slid_blurk, 'BLUR', 1, int(max([height, width]) / 20), valinit=5)
        self.slid_blurk.on_changed(self.go_thresholding_update)
        self.go_thresholding_update(0)
        plt.show()

    def go_thresholding_update(self, value):
        th = int(self.slid_th.val)
        blur = int(self.slid_blurk.val)
        if blur % 2 == 0:
            if blur - 1 < 1:
                blur = blur + 1
            else:
                blur = blur - 1
        self.imthresh = pycv.thresh(pycv.blur(self.im2, blur), th)
        self.im2_canvas = pycv.rgb(self.imthresh.copy())
        self.im1_canvas = self.im1.copy()
        # contours detect
        self.cnts = pycv.contours(self.imthresh, cv2.CHAIN_APPROX_NONE)
        # Draw Contours and Ellipses
        self.im1_canvas = pycv.draw(self.im1_canvas, self.cnts, [('contour', (255, 255, 255))])  # links
        self.im2_canvas = pycv.draw(self.im2_canvas, self.cnts,
                                    [('contour', (0, 255, 255)), ('fit', (0, 255, 0))])  # rechts
        self.thresh_ims[0].set_data(self.im1_canvas)
        self.thresh_ims[1].set_data(self.im2_canvas)
        self.btnResult.setEnabled(True)
        self.btnExport.setEnabled(True)

    # Result & Export
    def go_result(self):
        mpp = float(self.txtMPP.text())  # micrometers per pixel
        self.data = pycv.evaluate(self.cnts, mpp)
        elements = []
        if self.chbContours.isChecked():
            elements.append(('contour', (255, 255, 255)))
        if self.chbFitting.isChecked():
            elements.append(('fit', (0, 255, 0)))
        if self.chbIndex.isChecked():
            elements.append(('index', (255, 255, 255)))
        if self.chbAngle.isChecked():
            elements.append(('angle', (255, 0, 0)))
        self.imres = pycv.draw(self.im1, self.cnts, elements)
        pycv.shown([self.imres], True, True, 0, 0,True, 'Resulting Image')


    def go_export(self):
        # EXCEL
        if self.chbExcel.isChecked():
            try:
                if self.txtExcel.text() == '':
                    filename = self.txtImgOriginal.text() + '.xlsx'
                else:
                    filename = self.txtExcel.text()

                self.data.to_excel(filename)
                print('Excel-Export: {}'.format(filename))
                os.startfile(filename)
            except:
                print('Excel-Export failed! ({})'.format(filename))
        # EVALUATION IMAGE
        if self.chbEval.isChecked():
            try:
                if self.txtEval.text() == '':
                    filename = self.txtImgOriginal.text()
                    filename = os.path.splitext(filename)[0] + '-result.tif'
                else:
                    filename = self.txtEval.text()
                cv2.imwrite(filename,self.imres)
                print('Evaluate-Image-Export: {}'.format(filename))
            except:
                print('Evaluate-Image-Export failed! ({})'.format(filename))

if __name__ == "__main__":
    ek = Eichkoernchen()
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Eichkoernchen()
    ui.setupUi(MainWindow)
    ui.setupSignals()
    # SIGNAL HOOKS
    # Loading

    MainWindow.show()
    sys.exit(app.exec_())

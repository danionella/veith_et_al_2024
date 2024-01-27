from PyQt5.QtGui import QDoubleValidator, QIntValidator,QPalette,QColor
from PyQt5.QtWidgets import (QApplication,
                             QFileDialog, QMainWindow)
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,uic
from PyQt5.QtCore import QTimer,QDateTime
import sys
import logging
from vocloc_recorder import recorder
import json
import schedule
import numpy as np
from queue import Empty, Queue, Full
logger=logging.getLogger(__name__)





class main_ui(QtWidgets.QMainWindow):
    def __init__(self,confpth=None):
        super(main_ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('main.ui', self) # Load the .ui file
        self.show() # Show the GUI
        if confpth is None:
            confpth = ".\\config.json"
        with open(confpth) as f:
            conf = json.load(f)

        self.recorder=recorder(confpth ,conf['general']['savePath'])
        self.recorder.p_toplevel = conf['general']['savePath']
        self.recorder.setup(self.recorder.conf)
        # TODO check whether this is real copy or just reference
        self.conf = self.recorder.conf
        self.init_ui()
        self.update_fields()
        self.path_edit.setText(conf['general']['savePath'])
        self.start_time = QDateTime.currentDateTime()

        self.current_timer_function=None
        self.scheduled_jobs=[]
        #timeDisplay = time.toString('yyyy-MM-dd hh:mm:ss dddd')



    def check_scheduletime(self):
        if QDateTime.currentDateTime() > self.schedule_time_edit.dateTime():
            self.start_recording()



    def check_stoptime(self):
        t=QDateTime.currentDateTime()
        if self.start_time.msecsTo(t)/1000>float(self.time_edit.text()):
            self.stop_recording()


    def update_fields(self):
        self.conf['rec_name']=self.name_edit.text()
        self.recorder.p_toplevel=self.path_edit.text()
        self.recorder.setup(self.recorder.conf)




    def gload_folder(self):
        try:
            fname = QFileDialog.getExistingDirectory(self, 'Open folder',
                                                'C:\\', QFileDialog.ShowDirsOnly)
            logger.debug("Fnames:"+fname)
            self.path_edit.setText(fname)
            self.p_toplevel=self.path_edit.text()
            #self.stimulus_name.setText(self.mic.stim.name)
            #self.nameLineEdit.setText(self.mic.stim.name)
        except TypeError:
            pass

    def gload_config(self):

        fname= QFileDialog.getOpenFileName(self, 'Open file',
                                            '.\\', "Config_File (*.json)")
        logger.debug("Fnames:"+fname[0])
        logger.debug(fname)
        logger.debug(self.path_edit.text())
        with open(fname[0]) as f:
            self.recorder.conf = json.load(f)
        self.recorder.p_toplevel =  self.path_edit.text()
        self.recorder.setup(self.recorder.conf)
        # TODO check whether this is real copy or just reference
        self.conf = self.recorder.conf
        self.update_fields()


        #self.stimulus_name.setText(self.mic.stim.name)
        #self.nameLineEdit.setText(self.mic.stim.name)

    def start_recording(self):
        self.update_fields()
        self.recorder.start_recording(self.conf['rec_name'])
        self.start_time = QDateTime.currentDateTime()
        self.timer.stop()
        if self.current_timer_function is not None:
            self.timer.timeout.disconnect(self.current_timer_function)
        self.current_timer_function=self.timer.timeout.connect(self.check_stoptime)
        self.timer.start(5000)

    def start_schedule(self):
        logger.info("Scheduled!")
        self.timer.stop()
        if self.current_timer_function is not None:
            self.timer.timeout.disconnect(self.current_timer_function)
        self.current_timer_function=self.timer.timeout.connect(self.check_scheduletime)
        self.timer.start(5000)



    def stop_recording(self):
        self.timer.stop()
        self.recorder.stop_recording()

    def init_ui(self):
        """ Link all Events, Buttons and Slots

        Returns
        -------
        """

        self.timer = QTimer()

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_preview)
        self.video_timer.start(100)

        self.start_button.clicked.connect(self.start_recording)
        self.abort_button.clicked.connect(self.stop_recording)
        self.schedule_button.clicked.connect(self.start_schedule)
        self.preview_button.clicked.connect(lambda: self.recorder.start_recording('Preview', allow_saving=False))
        self.path_button.clicked.connect(self.gload_folder)
        self.config_button.clicked.connect(self.gload_config)

        self.time_edit.textChanged.connect(self.update_fields)
        self.name_edit.textChanged.connect(self.update_fields)
        self.path_edit.textChanged.connect(self.update_fields)

        self.schedule_time_edit.setDateTime(QDateTime.currentDateTime())



    def update_preview(self):
        try:
            t, idx, im = self.recorder.q_prev.get(timeout=0.5)
            self.video_preview.setImage(im.T)
        except Empty:
            pass



if __name__ == '__main__':



    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    dark_palette = QPalette()

    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    app.setPalette(dark_palette)

    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

    ex = main_ui(sys.argv[1])
    app.exec_()

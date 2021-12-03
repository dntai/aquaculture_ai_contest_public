import sys
import threading

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *

class QDash(QtCore.QObject):
    def __init__ (self, parent=None, **kwargs):
        super().__init__(parent)
        self.params = {**kwargs}
        self.run_cfg = dict(host='0.0.0.0', port = 8050)
        self.run_cfg.update(**self.params.get('run_cfg', {}))

        self.__app__ = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.layout = html.Div([html.H1('Hello, Dash World!')])
        pass  # __init__

    @property
    def app (self):
        return self.__app__

    def run(self, **kwargs):
        self.run_cfg.update(**kwargs)
        threading.Thread(target=self.app.run_server, kwargs=self.run_cfg, daemon=True).start()
    pass  # QDash

class ConsoleStream(QObject):
    """Redirects console output to text widget."""
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    pass # ConsoleStream

class ColorWidget(QWidget):
    def __init__(self, color, *args, **kwargs):
        super(ColorWidget, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)
        pass

    pass # Color

import sys, os
import threading

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import webbrowser

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *

__all__ = ['QDashApp', 'QDashWindow', 'dash_qt', 'dash_web', 'dash_main']

class QDashApp(QtCore.QObject):
    def __init__ (self, parent=None, app = None, popup_web = True, join_main = True, **kwargs):
        super().__init__(parent)
        self.params = {**kwargs}
        self.run_cfg = dict(host='0.0.0.0', port = 8050)
        self.run_cfg.update(**self.params.get('run_cfg', {}))
        self.global_scope = kwargs.get("global_scope", globals())
        self.thread = None
        self.popup_web = popup_web
        self.join_main = join_main

        if app is None:
            self.__app__ = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)
            self.app.layout = html.Div([html.H1('Hello, Dash World!')])
        else:
            self.__app__ = app
        pass  # __init__

    @property
    def app(self):
        return self.__app__

    def run(self, **kwargs):
        self.run_cfg.update(**kwargs)
        self.thread = threading.Thread(target=self.app.run_server, kwargs=self.run_cfg, daemon=True)
        self.thread.start()
        if self.popup_web:
            # webbrowser.open_new_tab(f'http://{self.run_cfg["host"]}:{self.run_cfg["port"]}')
            webbrowser.open_new(f'http://{self.run_cfg["host"]}:{self.run_cfg["port"]}')
        if self.join_main:
            while self.thread is not None and self.thread.isAlive(): self.thread.join(1)

    pass  # QDashApp

class QDashWindow(QMainWindow):
    def __init__ (self, parent=None, **kwargs):
        """ init """
        super().__init__(parent)

        # init window
        self.global_scope = kwargs.get("global_scope", globals())
        self.icon_path    = kwargs.get('icon_path', self.global_scope.get('icon_path'))
        self.window_title = kwargs.get('window_title', self.global_scope.get('window_title', 'Hello, Dash App!'))
        self.window_size  = kwargs.get('window_size', self.global_scope.get('window_size', (1024, 768)))
        self.app          = kwargs.get("app", self.global_scope.get('app'))

        # init dash app
        self.dash_app = QDashApp(app=self.app, global_scope=self.global_scope, popup_web = False, join_main = False)
        self.dash_app.run(debug=True, use_reloader=False)

        self.initializeUI()

        self.global_scope.update(**globals())
        pass # __init__

    def initializeUI(self):
        """ gui init """
        self.resize(self.window_size[0], self.window_size[1])

        # set icon
        if self.icon_path is not None and os.path.exists(self.icon_path) == True:
            app_icon = QIcon()
            app_icon.addPixmap(QPixmap(self.icon_path), QIcon.Selected, QIcon.On)
            self.setWindowIcon(app_icon)

        self.setWindowTitle(self.window_title)

        self.setupMainView() # Central View
        # self.setupToolsDockWidget()

        """ gui show """
        self.show()
        pass # initializeUI

    def setupMainView(self):
        """ main layout """
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        """ main widgets """
        self.browser  = QWebEngineView()
        self.browser.load(QUrl(f'http://{self.dash_app.run_cfg["host"]}:{self.dash_app.run_cfg["port"]}'))

        main_layout.addWidget(self.browser)

        """ main events """

        """ attach to central """
        self.setCentralWidget(main_widget)
        pass # setupMainView

    pass # QDashWindow

def dash_qt(app, global_scope = globals()):
    try:
        qapp = QApplication(sys.argv)
        qapp.setStyle(QStyleFactory.create("Fusion"))
        window = QDashWindow(app = app, global_scope = global_scope)
        sys.exit(qapp.exec_())
    except KeyboardInterrupt:
        sys.exit(0)
        pass
    pass # qt_main

def sigint_handler():
    pass

def dash_web(app, global_scope = globals()):
    try:
        qapp = QApplication(sys.argv)
        qapp.setStyle(QStyleFactory.create("Fusion"))
        dash_app = QDashApp(app = app, global_scope = globals())
        dash_app.run(debug=True, use_reloader=False, dev_tools_hot_reload=False, )
        sys.exit(qapp.exec_())
    except KeyboardInterrupt:
        sys.exit(0)
        pass
    pass  # web_main

def dash_main(app, global_scope = globals()):
    globals().update(**global_scope)
    try:
        app.run_server(host="0.0.0.0", debug=True, dev_tools_hot_reload=False, )
    except KeyboardInterrupt:
        sys.exit(0)
        pass
    pass # dash_main
from __future__ import absolute_import, division, print_function
from IPython import display, get_ipython
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except:
    pass

import sys, os
try:
    script_file = os.path.normpath(os.path.abspath(__file__))
    script_dir  = os.path.normpath(os.path.abspath(os.path.dirname(__file__))) # python
except:
    script_dir = os.path.normpath(os.path.abspath(".")) # jupyter-notebook
    script_file= f'{script_dir}/app.py'
root_dir    = os.path.normpath(os.path.abspath(script_dir + "/.."))
if root_dir in sys.path: sys.path.remove(root_dir)
sys.path.insert(1, root_dir)

from aquaculture.common import *
from aquaculture.apps.utils.qt_utils import *
from aquaculture.apps.dash_app import *

class MainWindow(QMainWindow):
    def __init__ (self, parent=None):
        """ init """
        super().__init__(parent)

        self.initializeUI()
        pass # __init__

    def initializeUI(self):
        """ gui init """
        self.resize(1024, 768)
        # set icon
        app_icon = QIcon()
        app_icon.addPixmap(QPixmap(f'{root_dir}/data/temps/icons/fish.png'), QIcon.Selected, QIcon.On)
        self.setWindowIcon(app_icon)
        self.setWindowTitle("Aquaculture Management")

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
        self.dash_app = MainDash(global_cfg = globals())
        self.dash_app.run(debug=True, use_reloader=False)
        self.browser.load(QUrl(f'http://{self.dash_app.run_cfg["host"]}:{self.dash_app.run_cfg["port"]}'))

        main_layout.addWidget(self.browser)

        """ main events """

        """ attach to central """
        self.setCentralWidget(main_widget)
        pass # setupMainView

    pass # MainWindow

def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create("Fusion"))
        window = MainWindow()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    pass # main

if __name__ == "__main__":
    main()
    pass
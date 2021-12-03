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

from aquaculture.apps.dash_app import *

def main ():
    try:
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create("Fusion"))
        dash_app = MainDash(global_cfg = globals())
        dash_app.run(debug=True, use_reloader=False)
        webbrowser.open_new_tab(f'http://{dash_app.run_cfg["host"]}:{dash_app.run_cfg["port"]}')
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    pass  # main

if __name__ == "__main__":
    main()
    pass
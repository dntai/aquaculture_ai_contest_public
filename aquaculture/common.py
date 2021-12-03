import os, sys
import threading

import matplotlib.pyplot as plt

import webbrowser

# Web Dash
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# App PyQT5
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets, QtChart
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtChart import *
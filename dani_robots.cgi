#!/usr/bin/python
import cgitb; cgitb.enable()  # This line enables CGI error reporting
from wsgiref.handlers import CGIHandler
import traceback
import os, sys, subprocess

home = os.path.expanduser("~")
sys.path.insert(0, home)

from __init__ import app

CGIHandler().run(app)


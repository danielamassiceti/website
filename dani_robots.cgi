#!/usr/bin/python
import cgitb; cgitb.enable()  # This line enables CGI error reporting
from wsgiref.handlers import CGIHandler
import traceback
import os, sys, subprocess

home = os.path.expanduser("~")
sys.path.insert(0, home)
#subprocess.call(['activate www_withpt'], executable='/bin/bash')

#activate_this = 'Software/anaconda2/envs/www_withpt/bin/activate'
#execfile(activate_this, dict(__file__=activate_this))

from __init__ import app

CGIHandler().run(app)


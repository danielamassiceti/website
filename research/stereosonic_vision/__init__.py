from flask import Blueprint, render_template

_stereosonic_vision = Blueprint('stereosonic_vision', __name__, template_folder='templates', static_folder='static', static_url_path='/static')
global cv, jobtitle, mykeywords

def set_global_params(cv_path, title, keywords):
    global cv, jobtitle, mykeywords
    cv = cv_path
    jobtitle = title
    mykeywords = keywords

@_stereosonic_vision.route('/')
def home():
    return render_template('stereosonic_vision.html', cv=cv, jobtitle=jobtitle, mykeywords=mykeywords)


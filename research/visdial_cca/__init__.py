from flask import Blueprint, render_template

_visdial_cca = Blueprint('visdial_cca', __name__, template_folder='templates', static_folder='static', static_url_path='/static')
global cv, jobtitle, mykeywords

def set_global_params(cv_path, title, keywords):
    global cv, jobtitle, mykeywords
    cv = cv_path
    jobtitle = title
    mykeywords = keywords

@_visdial_cca.route('/')
def home():
    return render_template('visdial_cca.html', cv=cv, jobtitle=jobtitle, mykeywords=mykeywords)


from flask import Blueprint, render_template

_geneval_visdial = Blueprint('geneval_visdial', __name__, template_folder='templates', static_folder='static', static_url_path='/static')
global cv, jobtitle, mykeywords

def set_global_params(cv_path, title, keywords):
    global cv, jobtitle, mykeywords
    cv = cv_path
    jobtitle = title
    mykeywords = keywords

@_geneval_visdial.route('/')
def home():
    return render_template('geneval_visdial.html', cv=cv, jobtitle=jobtitle, mykeywords=mykeywords)


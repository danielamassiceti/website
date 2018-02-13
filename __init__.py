import os
from flask import Flask, render_template
from research.rfs_vs_nns import _rfs_vs_nns
from research.bottom_up_top_down import _bottom_up_top_down
from research import stereosonic_vision as sv
from research import flipdial as fd

app = Flask(__name__)
app.template_folder = os.path.join(app.root_path, 'templates')

cv = 'danielamassiceti_cv_feb2018.pdf'
vd_host = '129.67.94.221'
vd_port = 16688

app.register_blueprint(_rfs_vs_nns, url_prefix='/research/rfs_vs_nns')
app.register_blueprint(_bottom_up_top_down, url_prefix='/research/bottom_up_top_down')

sv.set_cv(cv)
app.register_blueprint(sv._stereosonic_vision, url_prefix='/research/stereosonic_vision')

fd.set_global_params(cv, vd_host, vd_port)
app.register_blueprint(fd._flipdial, url_prefix='/research/flipdial')

@app.route('/')
@app.route('/home')
@app.route('/index')
def home():
    return render_template('index.html', cv=cv)

@app.route('/about')
def about():
    return render_template('about.html', cv=cv)

@app.route('/research')
def research():
    return render_template('research.html', cv=cv)

if __name__ == "__main__":
    app.run()

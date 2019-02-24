import os, config
from flask import Flask, render_template
from research.rfs_vs_nns import _rfs_vs_nns
from research.bottom_up_top_down import _bottom_up_top_down
from research import stereosonic_vision as sv
from research import flipdial as fd
from research import visdial_cca as vdc

app = Flask(__name__)
app.template_folder = os.path.join(app.root_path, 'templates')

app.register_blueprint(_rfs_vs_nns, url_prefix='/research/rfs_vs_nns')
app.register_blueprint(_bottom_up_top_down, url_prefix='/research/bottom_up_top_down')

sv.set_cv(config.cv)
app.register_blueprint(sv._stereosonic_vision, url_prefix='/research/stereosonic_vision')

fd.set_global_params(config.cv, config.vd_host, config.vd_port)
app.register_blueprint(fd._flipdial, url_prefix='/research/flipdial')

vdc.set_cv(config.cv)
app.register_blueprint(vdc._visdial_cca, url_prefix='/research/visdial_cca')

@app.route('/')
@app.route('/home')
@app.route('/index')
def home():
    return render_template('index.html', cv=config.cv)

@app.route('/about')
def about():
    return render_template('about.html', cv=config.cv)

@app.route('/research')
def research():
    return render_template('research.html', cv=config.cv)

if __name__ == "__main__":
    app.run()

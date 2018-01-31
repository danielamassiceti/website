from flask import Blueprint, request, render_template, url_for, redirect
import os, uuid, pickle, itertools
import visdial.visdial as visdial

global maxchatlen, chat, imglist, caps, vd, img_to_load, cv
maxchatlen = 10
chat = []
    
_flipdial = Blueprint('flipdial', __name__, template_folder='templates', static_folder='static', static_url_path='/static')

def set_cv(cv_path):
    global cv
    cv = cv_path

def initialise_model():
    global vd, caps, imglist
    imglist = os.listdir(os.path.join(_flipdial.static_folder, 'thumbnails'))
    caps = pickle.load(open(os.path.join(_flipdial.static_folder, 'cap_dict'), 'rb'))
    vd = visdial.VisDial(_flipdial.static_folder)
    print vd

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

@_flipdial.route('/', methods=['GET', 'POST'])
def home():
    return render_template('flipdial.html', cv=cv)

@_flipdial.route('/demo', methods=['GET', 'POST'])
def demo():
    global maxchatlen, chat, imglist, caps, img_to_load
    default_img = 'COCO_val2014_000000524382.jpg'
    
    if request.method=='POST' and len(chat) < maxchatlen :
        if 'reset' in request.form.keys():
            chat = []
            return render_template('flipdial_demo.html', chat=chat, scroll='demo', autofocus='autofocus', questiontext='enter your question...', imgs=imglist,
                    img_to_load=img_to_load, cv=cv)
            
        elif 'question' in request.form.keys():
            question = request.form['question']
            t = len(chat)*2

            if vd.check_img and vd.check_cap: #if img and cap features are loaded
                # get answer
                answer = vd.get_answer(t, question)
                answers = list(itertools.chain(*answer))
            else:
                return 'error'

            item = (question, answers)
            chat.append(item)
            print chat
            #img_to_load = ( url_for('.static', filename=os.path.join('images', default_img)), caps[default_img] )
            return render_template('flipdial_demo.html', chat=chat, scroll='demo', autofocus='autofocus', questiontext='', imgs=imglist,
                    img_to_load=img_to_load, cv=cv)
        elif 'img_to_load' in request.form.keys():
            
            # empty chat
            chat = []
            
            # get image selected
            imgname = request.form['img_to_load']
            img_path = os.path.join(_flipdial.static_folder, 'images', imgname)
            img_to_load = ( url_for('.static', filename=os.path.join('images', imgname)), '' )

            # set new image and caption features
            vd.reset(img_path, caps[imgname])

            return render_template('flipdial_demo.html', chat=chat, scroll='demo', questiontext='enter your question...', imgs=imglist, img_to_load=img_to_load, cv=cv)

        elif 'upload_img' in request.files.keys():
            
            # empty chat
            chat = []

            # get image uploaded and save locally with unique filename
            img_file = request.files['upload_img']
            unique_filename = str(uuid.uuid4())
            img_path = os.path.join(_flipdial.static_folder, 'uploaded_images', unique_filename)
            while os.path.exists( img_path ):
                unique_filename = str(uuid.uuid4())
                img_path = os.path.join(_flipdial.static_folder, 'uploaded_images', unique_filename)
            img_file.save(img_path)
            img_to_load = ( url_for('.static', filename=os.path.join('uploaded_images', unique_filename)), '' )

            # set new image and caption features
            vd.reset(img_path, 'PAD EOS')
            
            return render_template('flipdial_demo.html', chat=chat, scroll='demo', questiontext='enter your question...', imgs=imglist, img_to_load=img_to_load, cv=cv)
    else:
        
        # empty chat
        chat = []

        # set new image and caption features
        img_path = os.path.join(_flipdial.static_folder,'images', default_img)
        vd.reset(img_path, caps[default_img])
        
        img_to_load = ( url_for('.static', filename=os.path.join('images', default_img)), '' )
        return render_template('flipdial_demo.html', chat=chat, questiontext="enter your question...", imgs=imglist, img_to_load=img_to_load, cv=cv )

@_flipdial.route('/projects/flipdial/more_examples')
def moreexamples():
    return render_template('flipdial_moreexamples.html')

@_flipdial.route('/projects/flipdial/paper')
def paper():
    return render_template('arxiv.html')

@_flipdial.route('/projects/flipdial/code')
def code():
    return render_template('github.html')



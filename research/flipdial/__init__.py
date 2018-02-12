from flask import Blueprint, request, render_template, url_for, redirect
import os, uuid, pickle, itertools, requests, json

global maxchatlen, chat, imglist, caps, img_to_load, cv, current_img_path, current_cap
maxchatlen = 10
chat = []
    
_flipdial = Blueprint('flipdial', __name__, template_folder='templates', static_folder='static', static_url_path='/static')
imglist = os.listdir(os.path.join(_flipdial.static_folder, 'thumbnails'))
caps = pickle.load(open(os.path.join(_flipdial.static_folder, 'cap_dict'), 'rb'))

vd_server = 'http://129.67.94.221'
vd_port = '26736'

def set_cv(cv_path):
    global cv
    cv = cv_path

def get_vd_url(t, question, img_path, caption, chat):
    vd_url = vd_server + ':' + vd_port + '/get_answer?'
    vd_url += 't=' + str(t)
    vd_url += '&question=' + question
    vd_url += '&img_path=' + img_path
    vd_url += '&caption=' + caption
    vd_url += '&history=' + json.dumps(chat)
    return vd_url
    
def ravel_answers(answers):
    return answers.split('**')

@_flipdial.route('/', methods=['GET', 'POST'])
def home():
    return render_template('flipdial.html', cv=cv)

@_flipdial.route('/demo', methods=['GET', 'POST'])
def demo():
    global maxchatlen, chat, imglist, caps, img_to_load, cv, current_img_path, current_cap
    default_img = 'COCO_val2014_000000524382.jpg'

    if request.method=='POST' and len(chat) < maxchatlen :
        if 'reset' in request.form.keys():
            chat = []
            return render_template('flipdial_demo.html', chat=chat, scroll='demo', autofocus='autofocus', questiontext='enter your question...', imgs=imglist, img_to_load=img_to_load, cv=cv)
            
        elif 'question' in request.form.keys():
            question = request.form['question']
            t = len(chat)*2

            # get answer by pinging server where flipdial model is loaded
            vd_url = get_vd_url(t, question, img_path, caption, chat)
            try: 
                answer = requests.get(get_vd_url(t, question, img_path, caption, chat)) # get answer
                answers = ravel_answers(answer.text)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                return e

            item = (question, answers)
            chat.append(item)
            return render_template('flipdial_demo.html', chat=chat, scroll='demo', autofocus='autofocus', questiontext='', imgs=imglist, img_to_load=img_to_load, cv=cv)
        
        elif 'img_to_load' in request.form.keys(): # select image from panel
            
            # empty chat
            chat = []
            
            # get image selected
            imgname = request.form['img_to_load']
            current_cap = caps[imgname]
            current_img_path = os.path.join(_flipdial.static_folder, 'images', imgname)
            img_to_load = ( url_for('.static', filename=os.path.join('images', imgname)), '' )

            return render_template('flipdial_demo.html', chat=chat, scroll='demo', questiontext='enter your question...', imgs=imglist, img_to_load=img_to_load, cv=cv)

        elif 'upload_img' in request.files.keys(): # upload own image
            
            # empty chat
            chat = []

            # get image uploaded and save locally with unique filename
            img_file = request.files['upload_img']
            current_cap = 'PAD EOS'
            unique_filename = str(uuid.uuid4())
            current_img_path = os.path.join(_flipdial.static_folder, 'uploaded_images', unique_filename)
            while os.path.exists( current_img_path ):
                unique_filename = str(uuid.uuid4())
                current_img_path = os.path.join(_flipdial.static_folder, 'uploaded_images', unique_filename)
            img_file.save(current_img_path)
            img_to_load = ( url_for('.static', filename=os.path.join('uploaded_images', unique_filename)), '' )

            
            return render_template('flipdial_demo.html', chat=chat, scroll='demo', questiontext='enter your question...', imgs=imglist, img_to_load=img_to_load, cv=cv)
    else: #standard page load

        # empty chat
        chat = []
        
        # set image and caption to default options
        current_img_path = os.path.join(_flipdial.static_folder,'images', default_img)
        current_cap = caps[default_img]
        
        img_to_load = ( url_for('.static', filename=os.path.join('images', default_img)), '' )
        return render_template('flipdial_demo.html', chat=chat, questiontext="enter your question...", imgs=imglist, img_to_load=img_to_load, cv=cv )

@_flipdial.route('/more_examples')
def moreexamples():
    return render_template('flipdial_moreexamples.html')

@_flipdial.route('/paper')
def paper():
    return render_template('arxiv.html')

@_flipdial.route('/code')
def code():
    return render_template('github.html')



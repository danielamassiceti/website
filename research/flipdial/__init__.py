from flask import Blueprint, request, render_template, url_for, redirect
import os, uuid, pickle, itertools, requests, json

_flipdial = Blueprint('flipdial', __name__, template_folder='templates', static_folder='static', static_url_path='/static')

global params
global chat, img_to_load, current_img_path, current_cap

def set_global_params(cv_path, vd_host, vd_port):
    global params, chat
    params = {}
    params['maxchatlen'] = 10
    params['imglist'] = os.listdir(os.path.join(_flipdial.static_folder, 'thumbnails'))
    params['caps'] = pickle.load(open(os.path.join(_flipdial.static_folder, 'cap_dict'), 'rb'))
    params['cv'] = cv_path
    params['vd_host'] = vd_host
    params['vd_port'] = vd_port
    chat = []

def get_vd_url(t, question, img_path, caption, history):
    global params
    vd_url = 'http://' + params['vd_host'] + ':' + str(params['vd_port']) + '/get_answer?'
    vd_url += 't=' + str(t)
    vd_url += '&question=' + question
    vd_url += '&img_path=' + img_path
    vd_url += '&caption=' + caption
    vd_url += '&history=' + json.dumps(history)
    return vd_url
    
def ravel_answers(answers):
    return answers.split('**')

@_flipdial.route('/', methods=['GET', 'POST'])
def home():
    global params
    return render_template('flipdial.html', cv=params['cv'])

@_flipdial.route('/1vd_demo', methods=['GET', 'POST'])
def onevd_demo():
    global params
    global chat, img_to_load, current_img_path, current_cap
    default_img = 'COCO_val2014_000000524382.jpg'

    if request.method=='POST' and len(chat) < params['maxchatlen']:
        if 'reset' in request.form.keys():
            chat = []
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll='demo', autofocus='autofocus', questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
            
        elif 'question' in request.form.keys():
            question = request.form['question']
            t = len(chat)*2

            # get answer by pinging server where flipdial model is loaded
            if True:
            	vd_url = get_vd_url(t, question, current_img_path, current_cap, chat)
                print vd_url
                answer = requests.get(vd_url) # get answer
                print answer
                answers = ravel_answers(answer.text)
            #except requests.exceptions.RequestException as e:  # This is the correct syntax
            #    return e

            item = (question, answers)
            chat.append(item)
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll='demo', autofocus='autofocus', questiontext='', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
        
        elif 'img_to_load' in request.form.keys(): # select image from panel
            
            # empty chat
            chat = []
            
            # get image selected
            imgname = request.form['img_to_load']
            current_cap = params['caps'][imgname]
            current_img_path = os.path.join(_flipdial.static_folder, 'images', imgname)
            img_to_load = ( url_for('.static', filename=os.path.join('images', imgname)), '' )

            return render_template('flipdial_1vd_demo.html', chat=chat, scroll='demo', questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])

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

            
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll='demo', questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
    else: #standard page load

        # empty chat
        chat = []
        
        # set image and caption to default options
        current_img_path = os.path.join(_flipdial.static_folder,'images', default_img)
        current_cap = params['caps'][default_img]
        
        img_to_load = ( url_for('.static', filename=os.path.join('images', default_img)), '' )
        return render_template('flipdial_1vd_demo.html', chat=chat, questiontext="enter your question...", imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'] )

@_flipdial.route('/2vd_demo')
def twovd_demo():
    return render_template('flipdial_2vd_demo.html')

@_flipdial.route('/paper')
def paper():
    return redirect('https://arxiv.org/abs/1802.03803')

@_flipdial.route('/code')
def code():
    return render_template('github.html')



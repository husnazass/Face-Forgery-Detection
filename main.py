from flask import Flask, render_template, request, send_from_directory
from tflitelib import TensorflowLiteClassificationModel
import os
model = TensorflowLiteClassificationModel(model_path='static/model/model_tes.tflite',labels=['Fake','Real','Undetect'])
app = Flask(__name__)
app.secret_key = "projekakhir"
app.config['UPLOAD_FOLDER'] = 'static/img_check/'
@app.route("/")
def awal():
    return render_template('index.html')

@app.route("/freact",methods = ['POST','GET'])
def start():
    if request.method == 'POST':
        if request.files:
            img = request.files['image_input']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'],img.filename)
            img.save(img_path)
            label = model.run_from_filepath(os.path.join(app.config['UPLOAD_FOLDER'],img.filename))
            print(label)
            hasil = max([label[1][1],label[2][1]])
            if hasil==label[1][1]:
                return render_template('running.html',uploaded_image = 'img_check/'+img.filename,prediksi = label[1][0],prob = label[1][1])    
            elif hasil==label[2][1]:
                return render_template('running.html',uploaded_image = 'img_check/'+img.filename,prediksi = label[2][0],prob = label[2][1])
    elif request.method == 'GET':
        return render_template('running.html')
    return render_template('running.html')

@app.route('/contact_us',methods = ['POST','GET'])
def contact():
    return render_template('contact_us.html')
def send_uploaded_image(filename = ''):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__=="__main__":
    app.run(debug = True,port = '5002')
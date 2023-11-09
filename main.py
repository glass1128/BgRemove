from flask import Flask, render_template, request, make_response, jsonify, redirect, flash, send_from_directory
# from flask_jwt import JWT, jwt_required, current_identity
import json, os
from flask_cors import CORS, cross_origin
from remove_background_api import process_image

SECRET_KEY = "5713q3E#@f4h"
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, support_credentials=True)

@app.route('/api/image', methods=['POST'])
@cross_origin(supports_credentials=True)
def process_image_api():
    headers = request.headers

    # Print the headers
    for key, value in headers.items():
        print(f'{key}: {value}')
    transparent = request.form['transparent'] == "True"
    white_background = request.form['white_background'] == "True"
    passport = request.form['passport'] == "True"
    visualize = request.form['visualize'] == "True"

    req_file = None
    if not request.files == None and len(request.files) > 0:
        req_file = request.files['file']
    else:
        return {
            "is_success": False,
            "msg": "No file."
        }

    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "upload")
    file_path = os.path.join(file_path, req_file.filename)
    req_file.save(file_path)
    
    process_image(file_path, transparent, white_background, passport, visualize)
    
    base_name = os.path.basename(req_file.filename)
    file_name, _ = os.path.splitext(base_name)
    out_image_name = file_name + '.png'
    out_base_path = os.path.join(current_directory, "processed")
    transparent_path = os.path.join(os.path.join(out_base_path, "transparent"), req_file.filename)
    white_back_path = os.path.join(os.path.join(out_base_path, "white_background"), req_file.filename)
    passport_path = os.path.join(os.path.join(out_base_path, "passport"), req_file.filename)
    
    ret = {
        "is_success": True,
        "msg": "processed.",
        "transparent": False,
        "white_background": False,
        "passport": False
    }
    
    if transparent and os.path.exists(transparent_path):
        ret["transparent"] = True
    if white_background and os.path.exists(white_back_path):
        ret["white_background"] = True
    if passport and os.path.exists(passport_path):
        ret["passport"] = True

    return ret

@app.route('/download', methods=['GET'])
@cross_origin(supports_credentials=True)
def download_file():
    file_name = request.args.get('n')
    file_type = request.args.get('t')
    
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, "processed")
    folder_path = os.path.join(folder_path, file_type)
    
    return send_from_directory(folder_path, file_name)

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
from flask import Flask, render_template, Response,jsonify,request,session

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
import requests
import urllib.parse
from bs4 import BeautifulSoup


# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_detection import video_detection, img_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'amara'
app.config['UPLOAD_FOLDER'] = 'static/files'


#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")

def get_img_result(path_x = ''):
    return img_detection(path_x)
    

def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/stop', methods=['GET','POST'])
def stop():    
    session.clear()
    return home()
    
    
@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    form = UploadFileForm()
    img = None
    cls_result = None
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        session['img_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
        img ,cls_result= get_img_result(session['img_path'])
        
        if cls_result is not None:
            speciesname = cls_result[0]["name"].replace("_", " ").lower()
            session["species"] = speciesname
        else:
            session["species"] = None
        
            
    return render_template('index.html', form=form, imgname = img, clsresult = cls_result)



@app.route('/get_api_data')
def get_api_data():
    page_data = None
    if session.get('species', None) is not None:
        response = requests.get("https://api.inaturalist.org/v1/taxa?q="+ session.get('species', None) +"&order=desc&order_by=id")
        page_url = None
        if response.status_code == 200:
            page_url = response.json()["results"][0]["wikipedia_url"]
            print(page_url)
        else:
            print(f"Request failed with status code: {response.status_code}")
        
        
        if page_url is not None:   
            # Define the Wikipedia API endpoint
            api_url = "https://en.wikipedia.org/w/api.php"

            # Parse the URL to extract the page title
            parsed_url = urllib.parse.urlparse(page_url)
            page_title = urllib.parse.unquote(parsed_url.path[6:])  # Remove "/wiki/" from the path

            # Parameters for the API request
            params = {
                "action": "parse",
                "format": "json",
                "page": page_title,
                "prop": "text",
            }
            # Send the API request
            response = requests.get(api_url, params=params, allow_redirects=True)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()

                # Extract the page content in JSON
                page_data = data["parse"]["text"]["*"]
                
                print(page_data)
                soup = BeautifulSoup(page_data, 'html.parser')

                new_div = soup.new_tag('div',  **{'class': 'p-5 border border-3'})
                # Find the <table> tag
                table = soup.find('table')
                try:
                    consecutive_p = table.find_next('p')
                    new_div.append(consecutive_p)
                except:
                    pass
                
                # Remove the class and style attributes from the <table> tag
                if 'class' in table.attrs:
                    del table['class']
                if 'style' in table.attrs:
                    del table['style']
                
                for row in table.find_all('tr')[-2:]:
                    row.extract()
                    
                bootstrap_container = soup.new_tag('div', **{'class': 'container mx-auto'})
                # Create a new <div> row with Bootstrap classes
                bootstrap_row = soup.new_tag('div', **{'class': 'row mx-auto text-center'})

                # Find all <tr> tags within the table
                rows = table.find_all('tr')

                for row in rows:
                    # Create a new <div> col with Bootstrap classes
                    bootstrap_row2 = soup.new_tag('div', **{'class': 'row  text-center'})
                    
                    # Find all <td> tags within the <tr> tag
                    cells = row.find_all(['td', 'th'])

                    for cell in cells:
                        # # Create a new <p> tag to hold the cell content
                        # a_tags = cell.find_all('a')
                        # for a_tag in a_tags:
                        #     # Create a new <span> tag
                        #     span_tag = soup.new_tag('strong')
                        #     span_tag['style'] = 'display: inline-block;'
                        #     # Copy the text content of the <a> tag to the <span> tag
                        #     span_tag.string = a_tag.text
                        #     # Replace the <a> tag with the <span> tag
                        #     a_tag.replace_with(span_tag)
                            
                        for content in cell.contents:
                            bootstrap_col = soup.new_tag('div', **{'class': 'col'})
                            bootstrap_col.append(content)
                            bootstrap_row2.append(bootstrap_col)
                    
                    # Append the col to the row
                    bootstrap_row.append(bootstrap_row2)

                # Append the row to the container
                bootstrap_container.append(bootstrap_row)

                # Replace the table with the Bootstrap container
                table.replace_with(bootstrap_container)    
                # Append the contents of div1 and div2 to the new <div> tag
                new_div.append(bootstrap_container)
                
                page_data =new_div
                # print(page_data)
            else:
                print(f"Request failed with status code: {response.status_code}")
    return str(page_data)

# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET','POST'])
def webcam():
    session.clear()
    return render_template('webcam_page.html')

@app.route('/video_detection', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('video_page.html', form=form)

# @app.route('/img')
# def img():
#     #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
#     return Response(generate_img(path_x = session.get('img_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video')
def video():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
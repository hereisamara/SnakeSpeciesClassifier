# SnakeSpeciesClassifier

An application to detect snakes and classify their species in realtime. This is part of the project from CSC340 Artificial Intelligence course. 

# Data  
Dataset from SnakeClef2023 competition.

# Model
Yolov8 from https://github.com/ultralytics/ultralytics  

# How to run  
  
1. create virtual environment first  
`python -m venv myenv`  

2. activate myenv and install requirments.txt  
`pip install -r requirements.txt`  

3. Run flask app  
`python flaskapp.py`  

The trained models are not added to the project folder. First, add the models in the respective folders to be able to run the app.

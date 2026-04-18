from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,predictPipeline
application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictidata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
                gender = request.form.get('gender'),
                internet_access = request.form.get('internet_access'),
                study_environment = request.form.get('study_environment'),
                pass_fail = request.form.get('pass_fail'),

                age = int(request.form.get('age')),
                sleep_hours = float(request.form.get('sleep_hours')),
                social_media_hours = float(request.form.get('social_media_hours')),

                reading_score = float(request.form.get('reading_score')),
                writing_score = float(request.form.get('writing_score')),
                science_score = float(request.form.get('science_score')),
                final_exam_score = float(request.form.get('final_exam_score'))
            )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = predictPipeline()
        results=predict_pipeline.predict(pred_df)                
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
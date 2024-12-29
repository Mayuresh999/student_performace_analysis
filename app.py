import sys
from flask import Flask, request, render_template
from src.Student_Performance_Analyser.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.Student_Performance_Analyser.exception import CustomException


app = Flask(__name__, template_folder=r"templates")

@app.route('/')
def index():
    return render_template(r"index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template(r"home.html")
    
    else:
        try:
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity = request.form.get('race_ethnicity'),
                parental_level_of_education = request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course = request.form.get('test_preparation_course'),
                reading_score = int(request.form.get('reading_score')),
                writing_score = int(request.form.get('writing_score'))
            )
            data_df = data.get_data_as_df()
            print(data_df)
            pipeline = PredictPipeline()
            prediction = pipeline.predict(data_df)
            return render_template(r"home.html", results=prediction[0])
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)



# import os
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from fastapi import FastAPI, Request, Form
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# from typing import Optional
# from pydantic import BaseModel
# from src.Student_Performance_Analyser.pipeline.predict_pipeline import CustomData, PredictPipeline
# from src.Student_Performance_Analyser.exception import CustomException

# # Initialize FastAPI app
# app = FastAPI()

# # Mount templates and static files
# templates = Jinja2Templates(directory="templates")
# # app.mount("/static", StaticFiles(directory="static"), name="static")

# # Define input data model using Pydantic
# class StudentData(BaseModel):
#     gender: str
#     race_ethnicity: str
#     parental_level_of_education: str
#     lunch: str
#     test_preparation_course: str
#     reading_score: int
#     writing_score: int

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/predict", response_class=HTMLResponse)
# async def predict_form(request: Request):
#     return templates.TemplateResponse("home.html", {"request": request})

# @app.post("/predict")
# async def predict(
#     request: Request,
#     gender: str = Form(...),
#     race_ethnicity: str = Form(...),
#     parental_level_of_education: str = Form(...),
#     lunch: str = Form(...),
#     test_preparation_course: str = Form(...),
#     reading_score: int = Form(...),
#     writing_score: int = Form(...)
# ):
#     try:
#         data = CustomData(
#             gender=gender,
#             race_ethnicity=race_ethnicity,
#             parental_level_of_education=parental_level_of_education,
#             lunch=lunch,
#             test_preparation_course=test_preparation_course,
#             reading_score=reading_score,
#             writing_score=writing_score
#         )
        
#         data_df = data.get_data_as_df()
#         pipeline = PredictPipeline()
#         prediction = pipeline.predict(data_df)
        
#         return templates.TemplateResponse(
#             "home.html",
#             {
#                 "request": request,
#                 "results": prediction[0]
#             }
#         )
#     except Exception as e:
#         raise CustomException(e, sys)

# # API endpoint for programmatic access
# @app.post("/api/predict")
# async def predict_api(student_data: StudentData):
#     try:
#         data = CustomData(
#             gender=student_data.gender,
#             race_ethnicity=student_data.race_ethnicity,
#             parental_level_of_education=student_data.parental_level_of_education,
#             lunch=student_data.lunch,
#             test_preparation_course=student_data.test_preparation_course,
#             reading_score=student_data.reading_score,
#             writing_score=student_data.writing_score
#         )
        
#         data_df = data.get_data_as_df()
#         pipeline = PredictPipeline()
#         prediction = pipeline.predict(data_df)
        
#         return {"prediction": float(prediction[0])}
#     except Exception as e:
#         raise CustomException(e, sys)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
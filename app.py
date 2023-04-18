# Imports 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from flask import Flask, render_template, request

# Flask Setup 
app = Flask(__name__)

# Routes 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        df = pd.read_csv(file)
    
        for i in df.columns:
            if df[i].dtype == "object":
                df[i]=df[i].astype('category').cat.codes
        
        # Convert the DataFrame to a NumPy array
        X = df.drop(columns="math score").values
        y = df["math score"].values
        
        X_reading = df.drop(columns="reading score").values
        y_reading = df["reading score"].values
        
        X_writing = df.drop(columns="writing score").values
        y_writing = df["writing score"].values 
        
        # Resample the data to balance the classes
        rus = RandomUnderSampler(random_state=42)
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        X_rr, y_rr = ros.fit_resample(X_reading, y_reading)
        
        X_wr, y_wr = ros.fit_resample(X_writing, y_writing)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42)
        X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_rr, y_rr, random_state=42)
        X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_wr, y_wr, random_state=42)
        
        # Train the model
        math = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
        reading = DecisionTreeRegressor(random_state=42).fit(X_r_train, y_r_train)
        writing = DecisionTreeRegressor(random_state=42).fit(X_w_train, y_w_train)
        
        # Make a prediction using the test data
        m_pred = math.predict(X_test)
        r_pred = reading.predict(X_r_test)
        w_pred = writing.predict(X_w_test)
        
        # Calculate the R-squared score
        math_score = r2_score(y_test, m_pred)
        reading_score = r2_score(y_r_test, r_pred)
        writing_score = r2_score(y_w_test, w_pred)
        
        # Make a prediction using the uploaded data
        math_pred = math.predict(df.drop(columns="math score").values)
        reading_pred = reading.predict(df.drop(columns="reading score").values)
        writing_pred = writing.predict(df.drop(columns="writing score").values)
        
        # Display the scores on the webpage
        return render_template('index.html', df=df, math_pred=math_pred, reading_pred=reading_pred, writing_pred=writing_pred)
    else:
        return render_template('index.html')

    
 # Run the app
if __name__ == '__main__':
    app.run(debug=True)   

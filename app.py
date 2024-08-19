#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flask
import pandas as pd
import numpy as np
import pickle
import joblib

from flask import request,render_template, url_for
app = flask.Flask(__name__)
#app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

# main index page route
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=['POST'])
def predict():
    model = joblib.load(open("anemia_level_prediction_model.pkl", "rb"))
    if request.method == "POST":
        # Date_of_Journey
        age= request.form["v013"]
        region= request.form["v024"]
        has_radio= request.form["v120"]
        place_of_residence= request.form["v025"]
        educational_level= request.form["v106"]
        Religion= request.form["v130"]
        source_of_drinking_water= request.form["v113"]
        literacy= request.form["v155"]
        number_of_visit= request.form["v027"]
        toilet_facility= request.form["v116"]
        currently_working= request.form["v714"]
        current_marital_status= request.form["v501"]
        contraceptive_use_and_intension= request.form["v364"]
        currently_breastfeeding= request.form["v404"]
        body_mass_index= request.form["v445"]
        husband_education_level= request.form["v701"]
        husband_occupation= request.form["v705"]
        has_diarrhea= request.form["h11_1"]
        wealth_index= request.form["v190"]
        vitamin_A_in_last_6month= request.form["h34_1"] 
        type_of_cooking_fuel= request.form["v161"]
        weight_of_respondent= request.form["hw8_1"]
        
    final_arr = [age,region,has_radio,place_of_residence,educational_level,Religion,source_of_drinking_water,
                 literacy,number_of_visit,toilet_facility,currently_working,current_marital_status,contraceptive_use_and_intension,
                 currently_breastfeeding,body_mass_index,husband_education_level,
                 husband_occupation,has_diarrhea,wealth_index,vitamin_A_in_last_6month,type_of_cooking_fuel,weight_of_respondent] 
    print("final_arr:", final_arr)
    # Drop the rows with null values
    data = np.array(final_arr)
    data = data.reshape(1, -1)
    data = np.nan_to_num(data)
 # Make the prediction
    prediction = model.predict(data)
    # Return the appropriate response
    if prediction[0] == 1:
        return render_template("index.html", prediction_text=  "The Anemia level of Neonatal Women is Moderate")
    elif prediction[0] == 2:
        return render_template("index.html", prediction_text=  "The Anemia level of Neonatal Women is Mild")
    elif prediction[0] == 3:
        return render_template("index.html", prediction_text=  "The Anemia level of Neonatal Women is Non anemic") 
    else:
        return render_template("index.html", prediction_text=  "The Anemia level of Neonatal Women is Sever")
if __name__== "main":
    app.run(debug=False)




# In[ ]:





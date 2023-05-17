from flask import Flask,request
from flask_jsonpify import jsonify
from random import randint
from flask_cors import CORS
import db
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64

out=""

app = Flask(__name__)
CORS(app)


def token():
    tok=""
    for i in range(10):
        tok+=str(randint(0,9))
    return tok

#sample test api
@app.route('/')
def flask_mongodb_atlas():
    print("Called")
    return "flask mongodb atlas!"

#sample test api
@app.route('/plot')
def plot():
    img = BytesIO()
    y = [1,2,3,4,5]
    x = [0,2,1,3,4]

    plt.plot(x,y)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    url="data:image/png;base64,"+str(plot_url)
    return jsonify({
        "status":"valid",
        "url":url
        })


#api for verification
@app.route('/checkAuth',methods=['POST'])
def checkAuth():
    print("verifyauth")
    data = request.get_json()
    #print(data)    
    try:
        users=db.db.userCollection.find({"email": data["Email"],"token":data["token"]}) 
        output = [{'Email' : user['email']} for user in users] 
        #print(output)
        if len(output)==0:
            #print("valid")
            return jsonify({"status":"Valid"})
        else:
            #print("invalid")
            return jsonify({"status":"Notvalid"})  
    except Exception as e:
        print(e)
        return jsonify({"status":"Notvalid"})  

   

#user creation >>sign up api
@app.route('/createuser', methods=['POST'])
def createUser():
    request_data = request.get_json()
    print(request_data)
    users=db.db.userCollection.find({"email": request_data["email"]})
    output = [{'Name' : user['name'], 'Email' : user['email']} for user in users]
    print(output)
    try:
        if len(output) > 0:
            return jsonify({"status":"EMail Already Exist"})
        else:            
            db.db.userCollection.insert_one(request_data) 
            print("created successfully")
            return jsonify({"status":"user created Successfully"})  
            
    except Exception as e:
        print(e)
        return jsonify({"status":"Server Error"})  

#api for login
@app.route('/auth',methods=['POST'])
def read():
    try:
        request_data = request.get_json()
        print(request_data)
        users = db.db.userCollection.find({"email":request_data["email"],"password":request_data["password"]})
        output = [{'name' : user['name'],'Email' : user['email'],'token':user['token']} for user in users]
        print(output)    
        if(len(output)==1):
            tok=token()
            print(tok)
            filt={"email":output[0]['Email']}
            updat = {"$set": {'token' : tok}}
            print(filt,updat)
            db.db.userCollection.update_one(filt,updat)
            print("Updated")
            output[0]['token']=tok            
            return jsonify({"status":"Verified","data":output})   
        else:
            return jsonify({"status":"Invalid credential"}) 
            
    except Exception as e:
        print(e)
        return jsonify({"status":"Server Error"})  

#api for upload csv
@app.route('/uploadcsv',methods=['POST'])
def uploadCsv():
    print("Called")
    try:
        file=request.files.get('file')
        ftrain = pd.read_csv(file)
        ftrain['SalesPerCustomer'] = ftrain['Sales']/ftrain['Customers']
        ftrain['SalesPerCustomer'].head()
        # ftrain.head(5)
        ftrain.describe()
        targetPredictColumns=["Sales","Customers","SalesPerCustomer"]
        print(file)
        # print(hello(file))
        return jsonify({
            "status":"Uploaded Successfully!..",
            "columns":targetPredictColumns
        }) 
    except Exception as e:
        print(e)
        return jsonify({"status":"Internal Server Error"}) 



#api for prediction
@app.route('/predict',methods=['POST'])
def forecast_predict():
    global out
    try:
        from prophet import Prophet
        file=request.files.get('file')
        targetColumn=request.form.get('columnPredict')
        dayPredict=request.form.get('dayPredict')
        print(targetColumn,dayPredict)
        print(type(targetColumn),type(dayPredict))
        train = pd.read_csv(file)
        train.head(5)
        train.describe()
        #EDA PART************

        train.isnull().sum()
        train = train.drop(["Open","Promo"], axis = 1) #axis=1 for column
        train.info()
        #Normalization PART*******
        x_train=train.copy()
        x_train=x_train.drop(["Date","StateHoliday"], axis = 1)
        # data normalization with sklearn
        from sklearn.preprocessing import MinMaxScaler

        # fit scaler on training data
        norm = MinMaxScaler().fit(x_train)

        # transform training data
        X_train_norm = norm.transform(x_train)

        # standardization with  sklearn
        from sklearn.preprocessing import StandardScaler

        #x_train

        # apply standardization on numerical features
        for i in x_train:            
            # fit on training data column
            scale = StandardScaler().fit(x_train[[i]])
            
            # transform the training data column
            x_train[i] = scale.transform(x_train[[i]])
        
        # train.head(5)

        train['SalesPerCustomer'] = train['Sales']/train['Customers']
        train['SalesPerCustomer'].head()   
        train = train.dropna()

        #print("CK1")
        sales = train[train.Store == 1].loc[:, ['Date', targetColumn]]
        #print("CK2")

        # reverse to the order: from 2014 to 2015
        sales = sales.sort_index(ascending = False)

        sales['Date'] = pd.DatetimeIndex(sales['Date'])
        sales.dtypes
        #print("CK3")
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(sales[targetColumn])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        #print("CK34")

        sales = sales.rename(columns = {'Date': 'ds',targetColumn: 'y'})
        #print("CK4")
        sales_prophet = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)
        sales_prophet.fit(sales)
        #print("CK5")
        # Make a future dataframe for 6weeks
        sales_forecast = sales_prophet.make_future_dataframe(periods=int(dayPredict), freq='D')
        # Make predictions
        sales_forecast = sales_prophet.predict(sales_forecast)
        
        #print("CK6")
        #RMSE PART 
        # report performance
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(sales_forecast['daily'], sales_forecast['trend']))
        print('\nRMSE: %.3f' % rmse)

        #MAPE part
        mape=np.sum(sales_forecast['trend'])/np.sqrt(mean_squared_error(sales_forecast['daily'], sales_forecast['trend'])) *100
        print('\nMAPE: %.3f' % mape)

        print(len(sales_forecast))        
        #print("CK7")
        # matplotlib parameters
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
        img = BytesIO()
        sales_prophet.plot(sales_forecast, xlabel = 'Date', ylabel = targetColumn)
        # plt.title('Drug Store Forecasting',fontsize=18, color= 'green', fontweight='bold')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        url="data:image/png;base64,"+str(plot_url)
        #print("CK8")
        out = sales_forecast
        # out.to_csv("submission.csv")   
        val=randint(0,1000)
        out.to_csv("C:/Users/user/Desktop/"+str(val)+".csv")
        return jsonify({
            "status":"Uploaded Successfully!..",
            "graph":url,
            "ADF_Statistic":float(result[0]),
            "p_value":float(result[1]),
            "rmse":rmse,
            "mape":mape,
            "path":"C:/Users/user/Desktop/"+str(val)+".csv"
        }) 
    except Exception as e:
        print(e)
        return jsonify({"status":"Internal Server Error"}) 




if __name__ == '__main__':
    #print("hello")
    app.run(port=5002)
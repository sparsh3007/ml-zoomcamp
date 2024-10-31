
# Load the model

import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)


def predict(customer):

    X = dv.transform([customer])
    y_pred=model.predict_proba(X)[0,1]

    return round(y_pred,3)

customer1 = {"job": "management", "duration": 400, "poutcome": "success"}
print('Input customer:',customer1)
print('Subscription probability:', predict(customer1))

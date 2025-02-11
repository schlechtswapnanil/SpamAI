from django.shortcuts import render
from django.http import HttpResponse

import os
import joblib

modelRF= joblib.load(os.path.dirname(__file__) + "\\myModel_RF.pkl")
modelSVC= joblib.load(os.path.dirname(__file__) + "\\myModel_SVC.pkl")

# Create your views here.
def index(request):
    return render(request, 'index.html')

def checkspam(request):
    if(request.method == "POST"):
        
        algo=request.POST.get("algo")
        rawdata= request.POST.get("rawdata")
        final_answer="Not sure"
         
        if algo=='algo 1':
            final_answer=modelRF.predict([rawdata])[0]
        elif algo=='algo 2':
            final_answer=modelSVC.predict([rawdata])[0]
        param= {'answer': final_answer}
        
        print("!!!!ANSWER!!!======", final_answer)
        
        return render(request, 'output.html', param)
    else:
        return render(request, 'index.html')
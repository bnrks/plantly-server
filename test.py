import requests, json, glob
y_true, y_pred = [], []

for cls in ('healthy','powdery','rust'):
    for img_path in glob.glob(f'test/{cls}/*.jpg')[:15]:
        with open(img_path,'rb') as f:
            r = requests.post("https://learning-partially-rabbit.ngrok-free.app"+'/predict', files={'file':f})
        pred = r.json()['class']
        y_true.append(cls)
        y_pred.append(pred)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_true, y_pred, labels=['healthy','powdery','rust']))
print(classification_report(y_true, y_pred))

from ultralytics import YOLO
import cv2
import math
import json


def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    model=YOLO("../YOLO-Weights/best.pt")
    classification_model = YOLO("../YOLO-Weights/classification.pt")
    
    # print(model)
    
    while True:
        success, img = cap.read()
        results=model.predict(img,stream=True)
        print(results)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=r.names[0]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        yield img
    
def img_detection(path_x):

    snakeornot = True
    model=YOLO("../YOLO-Weights/best.pt")
    classification_model = YOLO("../YOLO-Weights/classification.pt")
    
    img = cv2.imread(path_x)
    results=model.predict(img,stream=True)
    for r in results:
        boxes=r.boxes
        snakeornot = False if len(boxes) == 0 else True
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=r.names[0]
            label=f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
    
    filename = "output.jpg"
    print("wrting")
    print(cv2.imwrite("static/outputs/"+filename, img))
    
    data = None
    if snakeornot:
        cls_results=classification_model.predict(img)
        data = []
        for r in cls_results:
            for c in range(0, 5):            
                id =r.probs.top5[c]
                item = {
                    "id": id,
                    "name": r.names[id],
                    "probability": str(round(float(r.probs.top5conf.cpu().numpy()[c])*100,2))+"%"
                }
                data.append(item)
    return (filename, data)

cv2.destroyAllWindows()
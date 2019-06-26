import numpy as nu
import matplotlib.pyplot as plt 
import cv2 as cv
from Detector import *
from sklearn.utils.linear_assignment_ import linear_assignment
from kalman import *
import copy
import warnings
import re
warnings.filterwarnings("ignore")

min_hits=1
max_age=4
tracker_list=[]
outputFile='YOLO_Kalman_hungarian.avi'

dict_class={"person":1,"car":2,"motorbike":3,"bicycle":4,"truck":5,"bus":6}

count_class=[0]*7

def iou_intersection(a, b):
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)

def hungarian_method(tracker,detector,iou_threshold=0.3):

	print("tracker length:",len(tracker),"detector length:",len(detector))
	iou_mat=np.zeros((len(tracker),len(detector)))
	for i in range(len(tracker)):
		for j in range(len(detector)):
			iou_mat[i,j]=iou_intersection(tracker[i],detector[j])

	matched_id=linear_assignment(-iou_mat)

	unmatched_tracker=[]
	unmatched_detector=[]

	actual_tracker=[x for x in range(len(tracker))]
	actual_detector=[x for x in range(len(detector))]

	for i in actual_tracker:
		if(i not in matched_id[:,0]):
			unmatched_tracker.append(i)

	for i in actual_detector:
		if i not in matched_id[:,1]:
			unmatched_detector.append(i)

	for m in matched_id:
		if(iou_mat[m[0],m[1]]<iou_threshold):
			unmatched_tracker.append(m[0])
			unmatched_detector.append(m[1])

	return [np.array(matched_id),np.array(unmatched_tracker),np.array(unmatched_detector)]

video='siemens 9 40 to 11 am 29th Jan.MP4'
cap = cv.VideoCapture(video)
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
OB=Object_Detector()
while cv.waitKey(1)<0:

	hasFrame, frame = cap.read()
	#cv.imshow("Frame",frame)
	if not hasFrame:
		print('Done Processing')
		cv.waitKey(3000)
		cap.release()
		break
	[detected_box,PredClass]=OB.localization(frame)
	YOLO_box=[]
	for i in detected_box:
		c=i[2]+i[0]
		d=i[3]+i[1]
		a=i[0]
		b=i[1]
		YOLO_box.append(copy.copy((a,b,c,d)))
	z_box=YOLO_box
	x_box=[]

	if(len(tracker_list)>0):
		for tkr in tracker_list:
			x_box.append(tkr.box)

	matched_id,unmatched_tracker,unmatched_detector=hungarian_method(x_box,z_box)

	if(matched_id.size>0):
		#predict and update kalman tracker
		for tracker_index, detector_index in matched_id:
			z=z_box[detector_index]
			z=np.expand_dims(z,axis=0).T
			tracker_object=tracker_list[tracker_index]
			tracker_object.kalman_filter(z)
			x=tracker_object.x_state.T[0].tolist()
			new_x=[x[0],x[2],x[4],x[6]]
			x_box[tracker_index]=new_x
			tracker_object.box=new_x
			tracker_object.hits+=1
			tracker_object.no_losses=0
			tracker_list[tracker_index]=tracker_object


	if(unmatched_detector.size>0):
		#tracker is not avilable so create a new tracker
		for trk in unmatched_detector:
			z=z_box[trk]
			class_name=PredClass[trk]
			temp_count=count_class[dict_class[class_name]]
			count_class[dict_class[class_name]]+=1
			z=np.expand_dims(z,axis=0).T
			new_tracker_object=Tracker()
			x=np.array([[z[0],0,z[1],0,z[2],0,z[3],0]]).T
			new_tracker_object.x_state=x
			new_tracker_object.predict_only()
			x=new_tracker_object.x_state.T[0].tolist()
			new_x=[x[0],x[2],x[4],x[6]]
			x_box.append(new_x)
			new_tracker_object.box=new_x
			new_tracker_object.id=class_name+str(':')+str(temp_count)
			tracker_list.append(new_tracker_object)

	if(unmatched_tracker.size>0):
		#object would have removed
		for trk in unmatched_tracker:
			tracker_removed=tracker_list[trk]
			tracker_removed.no_losses+=1
			tracker_removed.predict_only()
			x=tracker_removed.x_state.T[0].tolist()
			new_x=[x[0],x[2],x[4],x[6]]
			tracker_removed.box=new_x
			x_box[trk]=new_x
	good_tracker=[]
	for trk in tracker_list:
		if((trk.hits>=min_hits) and (trk.no_losses<=max_age)):
			good_tracker.append(trk)
			x_cv2=trk.box
			(x1,y1,x2,y2)=x_cv2
			cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
			cv.putText(frame,trk.id,(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
			cv.imshow("Frame",frame)
			vid_writer.write(frame.astype(np.uint8))
	for x in tracker_list:
		if(x.no_losses>max_age):
			print("object leaving:",x.id)
			temp=re.findall('[a-z]+',x.id)[0]
			count_class[dict_class[temp]]-=1
	tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
'''
	index=-1
	for box in detected_box:
		index+=1
		(x, y, w, h) = [int(v) for v in box]
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv.putText(frame,PredClass[index],(x,y),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
	cv.imshow("Frame", frame)
'''

import argparse
import cv2


def main():
    programArguments = argumentsParser()
    if programArguments['image']:
        print("image")
        readImageSource(programArguments)
    else:
        print("video")
        readVideoSource(programArguments)

def argumentsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
        help="path to the input image")
    ap.add_argument("-v", "--video",
        default=0,
        help="path to the input video. leave blank for web-cam")
    ap.add_argument("-s", "--save",
        default = '',
        help="")
    #    ap.add_argument("-c", "--cascade", nargs='+',
    #        default="haarcascade_frontalcatface.xml",
    #        help="path to haar cascade")

    return vars(ap.parse_args())


def detectCascade(imageColour,cascadeDef):
    imageGray = cv2.cvtColor(imageColour, cv2.COLOR_BGR2GRAY)
    detectedCascades = cascadeDef.detectMultiScale(imageGray, scaleFactor=1.2,
    minNeighbors=8, minSize=(10, 10))
    #print(str(cascadeDef)+str(len(detectedCascades)))
    return detectedCascades

def drawCascadesFrame(detectedCascades,imageColour,featureName,featureRGB):
    #imageColourCropped=[]
    if len(detectedCascades)>0:
        for (i,(x,y,w,h)) in enumerate(detectedCascades):
            #print(detectedCascades)
            cv2.rectangle(imageColour,(x,y),(x+w,y+h),featureRGB,2)
            cv2.putText(imageColour, featureName+":"+str(i+1), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, featureRGB, 2)
            #imageGrayCropped = gray[y:y+h, x:x+w]
            #imageColourCropped.append(imageColour[y:y+h, x:x+w])  
        #return imageColourCropped

def detectFaceFeatures(frame):
    face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('cascades\haarcascade_smile.xml')

    detectedFaceCascades = detectCascade(frame,face_cascade)
    drawCascadesFrame(detectedFaceCascades,frame,featureName='face',featureRGB=(0,125,0))
    for (i,(x,y,w,h)) in enumerate(detectedFaceCascades):
        frameCropped = frame[y:y+h, x:x+w]
        detectedSmileCascades = detectCascade(frameCropped,smile_cascade)
        detectedEyesCascades = detectCascade(frameCropped,eye_cascade)

        drawCascadesFrame(detectedSmileCascades,frameCropped,featureName='smile',featureRGB=(255,255,0))
        drawCascadesFrame(detectedEyesCascades,frameCropped,featureName='eye',featureRGB=(255,0,255))

def detectCatFeatures(frame):
    cat_face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalcatface.xml')
    eye_cascade = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')

    detectedCatFaceCascades = detectCascade(frame,cat_face_cascade)
    drawCascadesFrame(detectedCatFaceCascades,frame,featureName='cat',featureRGB=(169, 230, 138))
    for (i,(x,y,w,h)) in enumerate(detectedCatFaceCascades):
        frameCropped = frame[y:y+h, x:x+w]
        detectedCatEyesCascades = detectCascade(frameCropped,eye_cascade)
        drawCascadesFrame(detectedCatEyesCascades,frameCropped,featureName='cat_eye',featureRGB=(169, 230, 251))

def readVideoSource(programArguments):
    
    videoSource = programArguments['video']    
    #Open video source:
    videoCapture = cv2.VideoCapture(videoSource)
    videoCapture.set(3, 640) #WIDTH
    videoCapture.set(4, 480) #HEIGHT

    #Open video destination
    if programArguments['save']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 5.0, (640,480))

    print(type(videoCapture))

    cameraStatus, frame = videoCapture.read()

    while(cameraStatus):
        # Capture frame-by-frame
        cameraStatus, frame = videoCapture.read()
        #print(ret)
        detectFaceFeatures(frame)
        detectCatFeatures(frame)
        cv2.imshow('Video Capture',frame)
        if programArguments['save']:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
    cv2.destroyAllWindows()


def readImageSource(programArguments):
    imageSource = programArguments['image']
    image = cv2.imread(imageSource)
    detectFaceFeatures(image)
    detectCatFeatures(image)
    cv2.imshow("Image Capture", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
import cv2 as cv2
#from pyzbar.pyzbar import decode
import os
import re
import numpy as np
newname = "xxxxxxxxxxxxx"
from multiprocessing import Pool
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
#import ray
#img = cv2.imread("Picture1.jpg")

#print(decode(img, scan_locations=True))
def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

#ray.init()

#@ray.remote
def rename_file(filename):
    #make all of the below a function def - with input as only the filename.
    img = cv2.imread(filename)
    ori_img = cv2.imread(filename)
    qrCodeDetector = cv2.QRCodeDetector()
    oldfilename = filename
    #a = qrCodeDetector.detectAndDecode(img)
    #print (a[0])
    #splitted = a.split(",")
    #        for x in range(20):
    #            for y in range(10):
    #                alpha = x # Contrast control (1.0-3.0)
    #                beta = y # Brightness control (0-100)
    #                adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #                a = qrCodeDetector.detectAndDecode(adjusted)
    #                print (a[0])
    alpha = 10 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    a = qrCodeDetector.detectAndDecode(adjusted)
    b = a[0]
    skip = 0
    print (a[0] + "adjusted")
    if len(b)>1:
        skip = 1
    a = qrCodeDetector.detectAndDecode(img)
    c = a[0]
    print (c + "native")
    z = 0
    failed = 0
    
    alpha = 5 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    a = qrCodeDetector.detectAndDecode(adjusted)
    d = a[0]
    skip = 0
    data = ""
    ender = 0
    while True:
        if len(b)>1:
            print ("found with adjust")
            data = b
            ender = 1
            break
        b = ""
        if len(c)>1:
            print ("found native")
            data = c
            ender = 1
            break
        if len(d)>1:
            print ("found with adjust")
            data = d
            ender = 1
            break
        once = 0
        if len(data)<1:        
            b = ""
            z = z + 1
            rotated = rotate_image(img, z)
            a = qrCodeDetector.detectAndDecode(rotated)
            b = a[0]
            print (a[0] + "rotated by " + str(z))
            if len (b)>1:
                print ("found with rotate")
                data = b
                ender = 1
                break
            if z>90:
                print ("giving up rotating, will try contrast brightness matrix")
                failed = 1
                break
        else:
            break
    
    while True:
        if ender<1:
            for x in range(21):
                if len(data)>0:
                    break
                if ender>0:
                    break
                for y in range(11):
                    if len(data)>0:
                        break
                    alpha = x # Contrast control (1.0-3.0)
                    beta = y # Brightness control (0-100)
                    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    a = qrCodeDetector.detectAndDecode(adjusted)
                    #print (a[0])
                    print (str(x)+" " +str(y))
                    e = a[0]
                    if len(e)>1:
                        print ("found with aplpha betra matrix")
                        failed = 0
                        print (a[0])
                        data = e
                        ender = 1
                        once = 1
                        break
                    if y>9:
                        if x>19:
                            ender = 1
                            failed = 1
                            
        else:
            break
                    
        #print(data)
    
    
    if failed >0:
        newname = "failed_"+filename
        if newname in os.listdir("."):
           newname="duplicate_"+newname
        os.rename(filename, newname)
        print("gave up on this one - flagged as failed")
        print(newname)
        newname="xxxxxxxxxxxxx"
    
    if len (data)>0:
        print (str(data))
        datasplit = data.split(";")
        print (datasplit)
        last = datasplit[1]
        first = datasplit[0]
        newname =  first + "_rep" + last+ ".bmp"
        print (newname)
        #cv2.imwrite(newname, ori_img)
        if newname in os.listdir("."):
            newname="duplicate_"+newname
        os.rename(filename, newname)
        newname = "xxxxxxxxxxxxx"
        outputdata = oldfilename + "\t" + newname
        return outputdata

qrCodeDetector = cv2.QRCodeDetector()


#print(qrCodeDetector.detectAndDecode(img))
z= 0
list_of_bmps = ""
for filename in os.listdir("."):
    #print (filename)
    if filename.endswith("bmp"):
        #make a list of bmp filenames names here, then pass that to pool?
        list_of_bmps = list_of_bmps + filename + "\n"

print (list_of_bmps)
list_of_bmps_trim = list_of_bmps[:-1]
list_of_bmps_trim_split = list_of_bmps_trim.split("\n")
print (list_of_bmps_trim_split)
print ("\n\n\n")
        
#pool = Pool()

#for entry in list_of_bmps_trim_split:
#    rename_file(entry)
#    #u[p to here works]

def foo(arg1):
    '''
    body of the function
    '''
    output = str(arg1)
    #name = arg1+".test"
    t = open(str(arg1)+".test", "w")
    return output
#input = [11,32,44,55,23,0,100,...] # arbitrary list
num_cores = multiprocessing.cpu_count()
foo_ = partial(rename_file)
# arg1 is being fetched from input list
output = Parallel(n_jobs=num_cores)(delayed(foo_)(i) for i in list_of_bmps_trim_split)
print ("done")
#num_cores = multiprocessing.cpu_count()
#foo_ = partial(rename_file, arg2=arg2)
#output = Parallel(n_jobs=num_cores)(delayed(foo_)(i) for i in list_of_bmps_trim_split)
    
    #rename_file.remote(x)
    #y_id = solve2.remote(1)
#here can extract the names from the info and rename (or renamed to failed)



        
        #for z in range (50):
            #rotated = rotate_image(img, z)
            #a = qrCodeDetector.detectAndDecode(rotated)
            #print (a[0] + "rotated by " + str(z))
            #cv2.imwrite(filename+"processed.jpg", adjusted)
        #for x in range (100,2):
            #img = change_brightness(img, value=1)
            #a = qrCodeDetector.detectAndDecode(img)
            #print (a[0])



#adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
#for x in range(100):      
#    frame = increase_brightness("2.jpg", value=int(x))
#    a = qrCodeDetector.detectAndDecode(img2)
#    print (a[0])
        
#img = cv2.imread("2.jpg")
#img = change_brightness(img, value=100) #increases
#cv2.imwrite("2_bright.jpg", img)

#Import all the libraries
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import dlib
import os
import imageio

#Reading the images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

#Converting the image to gray scale
image1_gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

#Forming mask of image1 and image2
image1_mask = np.zeros_like(image1_gray)
image2_mask = np.zeros_like(image2_gray)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

landmarks_points_image1 = []
landmarks_points_image2 = []

num = int(input("Enter 1 for manual and 2 for automatic face detection : "))

if num == 1:
    fptr = open('TilePoints.txt','r')
    for j in fptr:
        a = j.split()
        landmarks_points_image1.append((int(a[0]), int(a[1])))
        landmarks_points_image2.append((int(a[2]),int(a[3])))
    fptr.close()
else:
    #Detecting landmark points of image1
    face1 = detector(image1_gray)
    for face in face1:
        landmarks = predictor(image1_gray,face)
        landmarks_points_image1 = []
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points_image1.append((x,y))
            
    #Append extremeties of image
    h = image1.shape[0]
    w = image1.shape[1]
    landmarks_points_image1.append((0,0))
    landmarks_points_image1.append((w-1,0))
    landmarks_points_image1.append((0,h-1))
    landmarks_points_image1.append((w-1,h-1))


    #Detecting landmark points of image2
    face2 = detector(image2_gray)

    for face in face2:
        landmarks = predictor(image2_gray,face)
        landmarks_points_image2 = []
        for n in  range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points_image2.append((x,y))
            
    #Append extremeties of image
    h = image2.shape[0]
    w = image2.shape[1]
    landmarks_points_image2.append((0,0))
    landmarks_points_image2.append((w-1,0))
    landmarks_points_image2.append((0,h-1))
    landmarks_points_image2.append((w-1,h-1))
    

#Defining the no of frames
no_of_frames = 100
frames = np.array([np.zeros_like(image1)]*no_of_frames)


#Finding the landmark points of morphed images
morphed_landmark_points = []

#Traversing for each frame
for i in range(1,no_of_frames):
    k = i/no_of_frames
    landmarks_points_imagek = []
    
    #Finding landmark points for morphed image k
    for j in range(len(landmarks_points_image1)):
        x1 = landmarks_points_image1[j][0]
        y1 = landmarks_points_image1[j][1]
        x2 = landmarks_points_image2[j][0]
        y2 = landmarks_points_image2[j][1]
        
        xk = (1-k)*x1 + k *x2
        yk = (1-k)*y1 + k *y2
        landmarks_points_imagek.append((xk,yk))
    morphed_landmark_points.append(landmarks_points_imagek)
morphed_landmark_points = np.array(morphed_landmark_points,np.int32)


#Delaunay Triangulation
#Finding the indexes of delaunay Triangles
rect = cv2.boundingRect(image1_gray)
rect
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(landmarks_points_image1)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles,dtype=np.int32)
points = np.array(landmarks_points_image1,np.int32)


#Finding the indexes of triangles in image1
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

indexes_triangles = []

for t in triangles:
    pt1 = (t[0],t[1])
    pt2 = (t[2],t[3])
    pt3 = (t[4],t[5])
    
    index_pt1 = np.where((points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)
    
    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)
    
    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)
    
    if(index_pt1 is not None and index_pt2 is not None and index_pt3 is not None):
        triangle = [index_pt1,index_pt2,index_pt3]
        indexes_triangles.append(triangle)
        
        
#for k frames
for k in range(len(morphed_landmark_points)):
    for triangle_index in indexes_triangles:
        
        #Finding the triangle points of image1
        tri1_pt1 = landmarks_points_image1[triangle_index[0]]
        tri1_pt2 = landmarks_points_image1[triangle_index[1]]
        tri1_pt3 = landmarks_points_image1[triangle_index[2]]
        
        #Finding the triangle points of image2
        tri2_pt1 = landmarks_points_image2[triangle_index[0]]
        tri2_pt2 = landmarks_points_image2[triangle_index[1]]
        tri2_pt3 = landmarks_points_image2[triangle_index[2]]
        
        #Finding the triangle points in morphed image
        tri_morph_pt1 = morphed_landmark_points[k][triangle_index[0]]
        tri_morph_pt2 = morphed_landmark_points[k][triangle_index[1]]
        tri_morph_pt3 = morphed_landmark_points[k][triangle_index[2]]
        
        #Finding bounding rectangle of tri1
        triangle1 = np.array([tri1_pt1,tri1_pt2,tri1_pt3],dtype=np.int32)
        rect1 = cv2.boundingRect(triangle1)
        (x,y,w,h) = rect1
        cropped_triangle1 = image1[y:y+h,x:x+w]
        cropped_tri1_mask = np.zeros((h,w),np.uint8)
        
        points1 = np.array([[tri1_pt1[0] - x, tri1_pt1[1] - y],
                        [tri1_pt2[0] - x, tri1_pt2[1] - y],
                        [tri1_pt3[0] - x, tri1_pt3[1] - y]],np.int32)
        
        cv2.fillConvexPoly(cropped_tri1_mask,points1,255)
        cropped_triangle1 = cv2.bitwise_and(cropped_triangle1,cropped_triangle1,mask=cropped_tri1_mask)
        
        
        #Finding bounding rectangle of tri2
        triangle2 = np.array([tri2_pt1,tri2_pt2,tri2_pt3],dtype=np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x,y,w,h) = rect2
        cropped_triangle2 = image2[y:y+h,x:x+w]
        cropped_tri2_mask = np.zeros((h,w),np.uint8)
        
        points2 = np.array([[tri2_pt1[0] - x, tri2_pt1[1] - y],
                        [tri2_pt2[0] - x, tri2_pt2[1] - y],
                        [tri2_pt3[0] - x, tri2_pt3[1] - y]],np.int32)
        cv2.fillConvexPoly(cropped_tri2_mask,points2,255)
        cropped_triangle2 = cv2.bitwise_and(cropped_triangle2,cropped_triangle2,mask=cropped_tri2_mask)
        
        #Finding bounding rectangle of morphed_tri
        triangle_morph = np.array([tri_morph_pt1,tri_morph_pt2,tri_morph_pt3],dtype=np.int32)
        rect_morph = cv2.boundingRect(triangle_morph)
        (x,y,w,h) = rect_morph
        
        points_morph = np.array([[tri_morph_pt1[0]-x, tri_morph_pt1[1]-y],
                                [tri_morph_pt2[0]-x, tri_morph_pt2[1]-y],
                                [tri_morph_pt3[0]-x,tri_morph_pt3[1]-y]],dtype=np.int32)
        
        #Affine Transform
        points1 = np.float32(points1)
        points2 = np.float32(points2)
        points_morph = np.float32(points_morph)
        
        affine1 = cv2.getAffineTransform(points1,points_morph)
        affine2 = cv2.getAffineTransform(points2,points_morph)
        
        warped_triangle1 = cv2.warpAffine(cropped_triangle1,affine1,(w,h))
        warped_triangle2 = cv2.warpAffine(cropped_triangle2,affine2,(w,h))
        
        #Forming final warped triangle
        p = (k+1)/no_of_frames
        warped_triangle_final  = (1-p)*warped_triangle1 + p*warped_triangle2
        warped_triangle_final = np.round_(warped_triangle_final,decimals=0,out=None)
            
        
        triangle_area = frames[k][y:y+h,x:x+w]
        warped_triangle_final = np.array(warped_triangle_final,dtype=np.uint8)
        triangle_area = np.array(triangle_area,dtype=np.uint8)
        
        triangle_area = cv2.add(triangle_area,warped_triangle_final)
        frames[k][y:y+h,x:x+w] = triangle_area 


src2 = './Gif'
if(os.path.exists(src2) == False):
    os.mkdir(src2)
    

#Writing images
src = './Images'
if(os.path.exists(src) == False):
    os.mkdir(src)

images = []
for i in range(no_of_frames-1):
    name = "img" + str(i+1) + ".jpg"
    cv2.imwrite(f'{src}/{name}',frames[i])
    file_path = os.path.join(src, name)
    images.append(imageio.imread(file_path))
    
imageio.mimsave(f'{src2}/output.gif', images,fps = no_of_frames/2)

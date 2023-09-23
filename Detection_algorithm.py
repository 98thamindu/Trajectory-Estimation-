import cv2
import mediapipe as mp
import pandas as pd
import statistics
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools

# create empty lists to store x, y, z coordinates of each landmark for all frames
x = [[] for _ in range(33)]
y = [[] for _ in range(33)]
z = [[] for _ in range(33)]

# create empty lists to store the speed of each landmark for all frames
cordinate_speed =[[] for _ in range(33)]

# create empty list to store the time stamp of each frame
time_matrix=[]

# create empty lists to store the angle of each knee and elbow for all frames
left_knee_angle=[]
right_knee_angle=[]
left_elbow_angle =[]
right_elbow_angle =[]
# create empty lists to store the angle of each knee and elbow for all frames
hip_shoulder=[]
right_eye_heel_left_eye_heel=[]
average_hip_knee_to_knee_ankle_ratio =[]
average_shoulder_hip_to_hip_ankle_ratio =[]
average_shoulder_elbow_to_elbow_wrist_ratio =[]
import pandas as pd
# Create an empty dictionary to store statistical data for each coordinate
data = {}
data1={}

# Iterate over each of the 32 coordinates and add keys for various statistical metrics to the dictionary
for i in range(33):
    # Define keys for mean and standard deviation of x, y, and z coordinates
    key = f'x{i} mean'
    key1 = f'y{i} mean'
    key2 = f'z{i} mean'
    key3 = f'x{i} standard deviation'
    key4 = f'y{i} standard deviation'
    key5 = f'z{i} standard deviation'
    
    # Define keys for mean and standard deviation of coordinate speed, left and right knee angle, and left and right elbow angle
    key6 = f'mean of cordinates speed{i}'
    key7 = f'standard deviation of cordinates speed{i}'
    key8 = f'mean of left knee{i}'
    key9 = f'mean of right knee{i}'
    key10 = f'mean of left elbow{i}'
    key11 = f'mean of right elbow{i}'
    key12 = f'standard deviation of left knee{i}'
    key13 = f'standard deviation of right knee{i}'
    key14 = f'standard deviation of left elbow{i}'
    key15 = f'standard deviation of right elbow{i}'
    
    
    # Initialize empty lists for each statistical metric key
    data[key] = []
    data[key1] = []
    data[key2] = []
    data[key3] = []
    data[key4] = []
    data[key5] = []
    data[key6] = []
    data[key7] = []
    data[key8] = []
    data[key9] = []
    data[key10] = []
    data[key11] = []
    data[key12] = []
    data[key13] = []
    data[key14] = []
    data[key15] = []
    
key16 = 'hip_shoulder'
key17 = 'right eye heel_left eye heel'
key18 = 'average hip knee to knee ankle ratio'
key19 = 'average shoulder hip to hip ankle ratio'
key20 = 'average shoulder elbow to elbow wrist ratio'
data1[key16] = []
data1[key17] = []
data1[key18] = []
data1[key19] = []
data1[key20] = []
# function to calculate angle using law of cosine
def angle(a,b,c):
    x=((a**2)+(b**2)-(c**2))/(2*a*b)
    ang_rad =math.acos(x)
    return math.degrees(ang_rad)

# function to calculate length of a line segment
def length(x,y,z,xp,yp,zp):
    leng=abs((((x-xp)**2)+((y-yp)**2))**0.5)
    return leng

# function to calculate the speed of a landmark between two frames
def speed(x,y,z,xp,yp,zp,pre_time,cur_time):
    dis=length(x,y,z,xp,yp,zp)
    sp=(dis/(cur_time-pre_time))
    return sp

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open the video capture device
cap = cv2.VideoCapture('Zig zag.mp4')
# Loop over the video frames
with pose:
    while cap.isOpened():
        # Read a frame from the video capture device
        success, image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the image to RGB and run the MediaPipe Pose model
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Print the coordinates of each landmark detected by the model
        if results.pose_landmarks is not None:
            current_time =time.perf_counter()
            time_matrix.append(current_time)
            for landmark_id in range(33):
                landmark = results.pose_landmarks.landmark[landmark_id]
                x[landmark_id].append(landmark.x)
                y[landmark_id].append(1-landmark.y)
                z[landmark_id].append(landmark.z)
            
        
        # Convert the image back to BGR and draw the landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Show the annotated image in a window
        cv2.imshow('MediaPipe Pose', image)
        
        
        # Exit the loop if the user presses the 'Esc' key
        if cv2.waitKey(5) & 0xFF == 27:
            break


# Calculate speed of each coordinate point for each frame
for i in range(33):
    for j in range(len(time_matrix)-1):
        # Calculate speed between consecutive frames
        cur_time = time_matrix[j+1]
        pre_time = time_matrix[j]
        speed_val = abs(speed(x[i][j+1],y[i][j+1],z[i][j+1],x[i][j],y[i][j],z[i][j],pre_time,cur_time))
        cordinate_speed[i].append(speed_val)

# Calculate angles of joints for each frame
for j in range(len(time_matrix)):
    # Calculate left knee angle for each frame
    a = length(x[24][j],y[24][j],z[24][j],x[26][j],y[26][j],z[26][j])
    b = length(x[28][j],y[28][j],z[28][j],x[26][j],y[26][j],z[26][j])
    c = length(x[24][j],y[24][j],z[24][j],x[28][j],y[28][j],z[28][j])
    left_knee_angle.append(angle(a,b,c))
    
    # Calculate right knee angle for each frame
    a = length(x[23][j],y[23][j],z[23][j],x[25][j],y[25][j],z[25][j])
    b = length(x[25][j],y[25][j],z[25][j],x[27][j],y[27][j],z[27][j])
    c = length(x[23][j],y[23][j],z[23][j],x[27][j],y[27][j],z[27][j])
    right_knee_angle.append(angle(a,b,c))
    
    # Calculate left elbow angle for each frame
    a = length(x[12][j],y[12][j],z[12][j],x[14][j],y[14][j],z[14][j])
    b = length(x[14][j],y[14][j],z[14][j],x[16][j],y[16][j],z[16][j])
    c = length(x[12][j],y[12][j],z[12][j],x[16][j],y[16][j],z[16][j])
    left_elbow_angle.append(angle(a,b,c))
    
    # Calculate right elbow angle for each frame
    a = length(x[11][j],y[11][j],z[11][j],x[13][j],y[13][j],z[13][j])
    b = length(x[13][j],y[13][j],z[13][j],x[15][j],y[15][j],z[15][j])
    c = length(x[11][j],y[11][j],z[11][j],x[15][j],y[15][j],z[15][j])
    right_elbow_angle.append(angle(a,b,c))

# Calculate angles of joints for each frame
for j in range(len(time_matrix)):
    # Calculate hip_shoulder for each frame
    a = length(x[24][j],y[24][j],z[24][j],x[23][j],y[23][j],z[23][j])
    b = length(x[11][j],y[11][j],z[11][j],x[12][j],y[12][j],z[12][j])
    
    hip_shoulder.append(a/b)
    
    # Calculate right eye heel_left eye heel for each frame
    a = length(x[5][j],y[5][j],z[5][j],x[30][j],y[30][j],z[30][j])
    b = length(x[2][j],y[2][j],z[2][j],x[29][j],y[29][j],z[29][j])
    
    right_eye_heel_left_eye_heel.append(a/b)
    
    # Calculate average hip knee to knee ankle ratio for each frame
    a = length(x[24][j],y[24][j],z[24][j],x[26][j],y[26][j],z[26][j])
    b = length(x[26][j],y[26][j],z[26][j],x[28][j],y[28][j],z[28][j])
    c = length(x[23][j],y[23][j],z[23][j],x[25][j],y[25][j],z[25][j])
    d = length(x[25][j],y[25][j],z[25][j],x[27][j],y[27][j],z[27][j])
    
    average_hip_knee_to_knee_ankle_ratio.append(((a/b)+(c/d))/2)
    
    # Calculate average shoulder hip to hip ankle ratio for each frame
    a = length(x[12][j],y[12][j],z[12][j],x[24][j],y[24][j],z[24][j])
    b = length(x[24][j],y[24][j],z[24][j],x[28][j],y[28][j],z[28][j])
    c = length(x[11][j],y[11][j],z[11][j],x[23][j],y[23][j],z[23][j])
    d = length(x[23][j],y[23][j],z[23][j],x[27][j],y[27][j],z[27][j])
    average_shoulder_hip_to_hip_ankle_ratio.append(((a/b)+(c/d))/2)
    
    # Calculate average shoulder elbow to elbow wrist ratio for each frame
    a = length(x[12][j],y[12][j],z[12][j],x[14][j],y[14][j],z[14][j])
    b = length(x[14][j],y[14][j],z[14][j],x[16][j],y[16][j],z[16][j])
    c = length(x[11][j],y[11][j],z[11][j],x[13][j],y[13][j],z[13][j])
    d = length(x[13][j],y[13][j],z[13][j],x[15][j],y[15][j],z[15][j])
    average_shoulder_elbow_to_elbow_wrist_ratio.append(((a/b)+(c/d))/2)

# Print statistics for each coordinate point and joint angle

    
print("hip_shoulder = ",statistics.mean(hip_shoulder))
data1['hip_shoulder'] = [statistics.mean(hip_shoulder)]
    
print("right eye heel_left eye heel = ",statistics.mean(right_eye_heel_left_eye_heel))
data1['right eye heel_left eye heel'] = statistics.mean(right_eye_heel_left_eye_heel)
    
print("average hip knee to knee ankle ratio = ",statistics.mean(average_hip_knee_to_knee_ankle_ratio))
data1['average hip knee to knee ankle ratio'] = statistics.mean(average_hip_knee_to_knee_ankle_ratio)
    
print("average shoulder hip to hip ankle ratio = ",statistics.mean(average_shoulder_hip_to_hip_ankle_ratio))
data1['average shoulder hip to hip ankle ratio'] = statistics.mean(average_shoulder_hip_to_hip_ankle_ratio)
    
print("average shoulder elbow to elbow wrist ratio =",statistics.mean(average_shoulder_elbow_to_elbow_wrist_ratio))
data1['average shoulder elbow to elbow wrist ratio'] = statistics.mean(average_shoulder_elbow_to_elbow_wrist_ratio)
    


# Print statistics for each coordinate point and joint angle
for i in range(33):
    
    print(f"Mean of x{i} = ",statistics.mean(x[i]))
    data[f'x{i} mean'] = [statistics.mean(x[i])]
    
    print(f"Mean of y{i} = ",statistics.mean(y[i]))
    data[f'y{i} mean'] = statistics.mean(y[i])
    
    print(f"Mean of z{i} = ",statistics.mean(z[i]))
    data[f'z{i} mean'] = statistics.mean(z[i])
    
    print(f"Standard Deviation of x{i} = ",statistics.stdev(x[i]))
    data[f'x{i} standard deviation'] = statistics.stdev(x[i])
    
    print(f"Standard Deviation of y{i} = ",statistics.stdev(y[i]))
    data[f'y{i} standard deviation'] = statistics.stdev(y[i])
    
    print(f"Standard Deviation of z{i} = ",statistics.stdev(z[i]))
    data[f'z{i} standard deviation'] = statistics.stdev(z[i])
    
    print(f"Average speed of cordinate {i} = ",statistics.mean(cordinate_speed[i]))
    data[f'mean of cordinates speed{i}'] = statistics.mean(cordinate_speed[i])
    
    print(f"Standard Deviation of the speed of cordinate {i} = ",statistics.stdev(cordinate_speed[i]))
    data[f'standard deviation of cordinates speed{i}'] = statistics.stdev(cordinate_speed[i])
    
    print("Mean of left knee angle = ",statistics.mean(left_knee_angle))
    data[f'mean of left knee{i}'] = statistics.mean(left_knee_angle)
    
    print("Mean of right knee angle = ",statistics.mean(right_knee_angle))
    data[f'mean of right knee{i}'] = statistics.mean(right_knee_angle)
    
    print("Mean of left elbow angle = ",statistics.mean(left_elbow_angle))
    data[f'mean of left elbow{i}'] = statistics.mean(left_elbow_angle)
    
    print("Mean of right elbow angle = ",statistics.mean(right_elbow_angle))
    data[f'mean of right elbow{i}'] = statistics.mean(right_elbow_angle)
    
    print("Standard Deviation of left knee angle = ",statistics.stdev(left_knee_angle))
    data[f'standard deviation of left knee{i}'] = statistics.stdev(left_knee_angle)
    
    print("Standard Deviation of right knee angle = ",statistics.stdev(right_knee_angle))
    data[f'standard deviation of right knee{i}'] = statistics.stdev(right_knee_angle)
    
    print("Standard Deviation of left elbow angle = ",statistics.stdev(left_elbow_angle))
    data[f'standard deviation of left elbow{i}'] = statistics.stdev(left_elbow_angle)
    
    print("Standard Deviation of right elbow angle = ",statistics.stdev(right_elbow_angle))
    data[f'standard deviation of right elbow{i}'] = statistics.stdev(right_elbow_angle)

l= len(y[29])
y_29=[]
y_30=[]
y_31=[]
y_32=[]
for i in range(l-2):
    if ((y[29][i+1]<=y[29][i])and((y[29][i+1]<=y[29][i+2]))):
        y_29.append(y[29][i+1])
for i in range(l-2):
    if ((y[30][i+1]<=y[30][i])and((y[30][i+1]<=y[30][i+2]))):
        y_30.append(y[30][i+1])
for i in range(l-2):
    if ((y[31][i+1]<=y[31][i])and((y[31][i+1]<=y[31][i+2]))):
        y_31.append(y[31][i+1])
for i in range(l-2):
    if ((y[32][i+1]<=y[32][i])and((y[32][i+1]<=y[32][i+2]))):
        y_32.append(y[32][i+1])


depth_point= (statistics.mean(y_31)+statistics.mean(y_32))/2
print("Depth point is = ",depth_point)
# Add depth_point value to the Excel file

data['land point'] = [depth_point]
data1['land point'] = [depth_point]

print("31 land point ",y[31])
print("32 land point ",y[32])
dp=[]
model1 = tf.keras.models.load_model('model.h5')
d1=model1.predict(np.array([0.1]))
d2=model1.predict(np.array([0.3]))
print("d1 is", d1)
print("d2 is", d2)
for d in range(len(y[31])):
    #new_input_data = np.array([(y[31][d] + y[32][d]) / 2])  # Wrap the value in a NumPy array
    #new_predictions = model1.predict(new_input_data)
    #dp.append(new_predictions)
    m=float((y[31][d] + y[32][d]) / 2)
    z = (((m - 0.1)/(0.3 - 0.1))*(6.75 - 4)) + 4
    dp.append(z)
    
print("Landmark depth =",dp)

print("31 x land point ",y[31])
print("32 x land point ",y[32])
dpx=[]
for d in range(len(x[31])):
    dpx.append(((x[31][d]+(x[32][d]))/2)*6)
    
print("Landmark depth x =",dpx)

print(len(dp))
print(len(dpx))


dp = np.array(dp).flatten()
# Fit a linear regression line (y = mx + b)
#coefficients = np.polyfit(dpx, dp, 1)
#m, b = coefficients

# Generate the line of best fit
#line_of_best_fit = [m * xi + b for xi in dpx]
coefficients = np.polyfit(dpx, dp, 9)
p = np.poly1d(coefficients)

# Generate the line of best fit (using the polynomial function)
line_of_best_fit = p(dpx)

angle_frame=[]
index=[]
coefficients1 = np.polyfit(dpx, dp, 9)
q = np.poly1d(coefficients1)
for d in range(len(x[31])):
    if d==len(x[31])-1:
        angle_frame.append(angle_frame[len(x[31])-2])
    else:
        tan_t= (q(dpx[d+1])-q(dpx[d]))/(dpx[d+1]-dpx[d])
        # Calculate the angle in radians
        angle_rad = math.atan(tan_t)

        # Convert radians to degrees
        angle_deg = math.degrees(angle_rad)
        angle_frame.append(angle_deg)
for d in range(len(x[31])):
    index.append(d+1)
print(len(angle_frame))
print(angle_frame) 

plt.plot(dpx, dp, color='blue', label='Observed Trajectory with noise data points')
# Plot the line of best fit
plt.plot(dpx, line_of_best_fit, color='red', label='Estimated trajectory')
#plt.axhline(y=4, color='green', label='Actual Trajectory of the person')
plt.xlim(0, 6)
plt.ylim(0, 10)

# Add labels and title
plt.xlabel('X Cordinates')
plt.ylabel('Y Cordinates')
plt.title('Trajectory of the person')
plt.legend()

# Display the plot
plt.show()

plt.figure()
plt.scatter(index, angle_frame, color='red', label='Estimated trajectory')
#plt.axhline(y=4, color='green', label='Actual Trajectory of the person')

# Add labels and title
plt.xlabel('Frame number')
plt.ylabel('Angle')
plt.title('Angle of the trajectory of the person')
plt.legend()
plt.show()
# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()

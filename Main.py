import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.fftpack import fft
from scipy.signal import find_peaks


#import the raw data for accelerometer, gyroscope, and labels
accelData = pd.read_csv('C:\\Users\\MichaelK\\Documents\\SeniorDesign\\accelerometer_data.txt')
gyroData = pd.read_csv('C:\\Users\\MichaelK\\Documents\\SeniorDesign\\gyroscope_data.txt')
labelData = pd.read_csv('C:\\Users\\MichaelK\\Documents\\SeniorDesign\\activity_data.txt', keep_default_na=False)


#combines all data into one file, correlating accel and gyro time values and applying a label to every entry
#inputs - 3 dataframes containing info from accelerometer, gyroscope, and activity
#output - dataframe containing combined info with correlated datapoints
def combine_data(accelData, gyroData, labelData):
    outData = pd.DataFrame(columns = ['Time','aX','aY','aZ','gX','gY','gZ','Activity'])
    #instantiating variables
    currentLabel = 'Null'
    gyroIndex = 0
    labelIndex = 0
    closeness = timedelta(seconds = 0.05)


    #Finding the current activity and saving it
    for index, row in accelData.iterrows():
        accelTime = datetime.strptime(row['Time'], '%m/%d/%y %H:%M:%S.%f') #using datetime objects rather than strings or floats
        oldLabel = currentLabel
        for i in range(labelIndex,labelData['Time'].count()-1):
            labelTime = datetime.strptime(labelData['Time'][i], '%m/%d/%y %H:%M:%S')
            #print(labelTime, accelTime)
            if labelTime < accelTime:
                labelIndex = i
                currentLabel = labelData['Activity'][i]
                if currentLabel == 'NULL' or currentLabel == 'Unanswered':
                    currentLabel = oldLabel
            else:
                break

        #test print -- uncomment this to ensure the function is working if it takes too long
        '''
        if oldLabel != currentLabel:
            print('Current Label: ' + currentLabel)
        '''
        oldLabel = currentLabel

        #correlating accel and gyro times
        gyroTime = datetime.strptime(gyroData['Time'][gyroIndex], '%m/%d/%y %H:%M:%S.%f')

        if gyroIndex < gyroData['Time'].count()-1: #Preventing out of bounds
            nextGyroTime = datetime.strptime(gyroData['Time'][gyroIndex + 1], '%m/%d/%y %H:%M:%S.%f')
            while abs(accelTime-gyroTime) > abs(accelTime-nextGyroTime):
                gyroIndex += 1
                if gyroIndex == gyroData['Time'].count()-1:
                    break
                gyroTime = datetime.strptime(gyroData['Time'][gyroIndex], '%m/%d/%y %H:%M:%S.%f')
                nextGyroTime = datetime.strptime(gyroData['Time'][gyroIndex + 1], '%m/%d/%y %H:%M:%S.%f')

        #putting all of the values together and appending them to the output dataframe
        outInfo = {'Time':accelTime, 'aX':row['X'],'aY':row['Y'],'aZ':row['Z'],'gX':gyroData['X'][gyroIndex],'gY':gyroData['Y'][gyroIndex],'gZ':gyroData['Z'][gyroIndex],'Activity':currentLabel}
        outData = outData.append(outInfo,ignore_index = True)
    return outData


#splits a dataframe into 1000 entry dataframes with 50% overlap
#inputs - df (dataframe)
#outputs - chunkList (list) of (dataframe)
def chunk(df): #input dataframe
    chunkCount = len(df.index)//1000 #determining the amount of adjacent chunks that will fit
    chunkList = []
    for i in range(chunkCount-1):
        tempChunk = df.iloc[i*1000:i*1000+1000] #normal chunks
        chunkList.append(tempChunk)
        if i*1000+1500 <= len(df.index):
            tempChunk = df.iloc[i*1000 - 500:i*1000+1500] #offset chunks
            chunkList.append(tempChunk)
    return chunkList


#removes rows from a dataframe containg outlying values in the specified column
#inputs - df (dataframe), col (string), devs (int)
#outputs - modified input dataframe
def remove_outliers_col(df,col, devs): #input dataframe and column name (string)
    std = df[col].std() #find standard deviation
    while df[col].max() > devs*std: #removing values higher than 5 standard deviations
        df = df.drop(df[col].idxmax())

    while df[col].min() < -devs*std: #removing values lower than 5 standard deviations
        df = df.drop(df[col].idxmin())
    df.head()
    return df #returns the modified dataframe


# In[6]:


#This function uses remove_outliers_col remove outliers from every column in a dataframe
#!!! do not use this on the entire data, use it on individual activities !!!
#inputs - df (dataframe) - devs (int) - devs is the amount of standard deviations to allow before a datapoint is removed
#outputs - modified input dataframe
def remove_outliers_df(df, devs):
    oldCount = len(df.index) #for determining percentage removed
    for key, value in df.iteritems(): #iterating over the dataframe
        if key != 'Time' and key != 'Activity':
            df = remove_outliers_col(df,key, devs) #calling remove outliers

    df = df.reset_index(drop=True) #fixing indexing, which has gaps after rows are removed
    print(df['Activity'].iloc[0],'datapoints removed:', oldCount-len(df.index)) #outputting removal statistics for each activity
    print(df['Activity'].iloc[0],'percentage removed: '+ str(round((oldCount-len(df.index))*100/oldCount,2))+'%\n')
    return df


#defining variables for feature extraction
t_n = 0.1
N = 1000
T = t_n / N
f_s = 1/T


#this function finds frequency from time for a chunk of data
def get_fft_values(chunk):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_ax_ = fft(chunk['aX'])
    fft_ay_ = fft(chunk['aY'])
    fft_az_ = fft(chunk['aZ'])
    fft_ax = 2.0/N * np.abs(fft_ax_[0:N//2])
    fft_ay = 2.0/N * np.abs(fft_ay_[0:N//2])
    fft_az = 2.0/N * np.abs(fft_az_[0:N//2])

    fft_gx_ = fft(chunk['gX'])
    fft_gy_ = fft(chunk['gY'])
    fft_gz_ = fft(chunk['gZ'])
    fft_gx = 2.0/N * np.abs(fft_gx_[0:N//2])
    fft_gy = 2.0/N * np.abs(fft_gy_[0:N//2])
    fft_gz = 2.0/N * np.abs(fft_gz_[0:N//2])

    outInfo = {'Frequency':f_values,'aX':fft_ax,'aY':fft_ay,'aZ':fft_az,'gX':fft_gx,'gY':fft_gy,'gZ':fft_gz,'Activity':chunk['Activity'][0:500]}
    outData = pd.DataFrame(outInfo)
    outData = outData.reset_index(drop=True)
    return outData


# In[32]:


def plot_axis(df, col):
    if col[0] == 'g': sensor = 'gyroscope '
    else: sensor = 'accelerometer '
    if col[1] == 'X': axis = 'X axis'
    elif col[1] == 'Y': axis = 'Y axis'
    else: axis = 'Z axis'
    plt.plot(df['Frequency'], df[col], linestyle='-', color='blue')
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.title("Frequency domain of the " + sensor + axis, fontsize=16)
    plt.show()
    return


# In[36]:


def autocorr(col):
    result = []
    result = np.correlate(col, col, mode='full')
    return result[len(result)//2:]


# In[37]:


def get_autocorr_values(df, T, N, f_s):
    for key, value in df.iteritems():
        if key != 'Time' and key != 'Activity':
            autocorr_values = autocorr(df[key])
            x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


# In[117]:


#function produced as a modification of code written by basj on stack overflow
#https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy

def peak_detection(df, col, cnt):
    x = df[col]
    prom = 10
    peaks2, _ = find_peaks(x, prominence=prom)
    while len(peaks2) < cnt:
        prom -= 0.005
        peaks2, _ = find_peaks(x, prominence=prom)
    plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['peaks'])
    plt.show()
    return


# In[29]:


allData = combine_data(accelData,gyroData,labelData)
allData.to_csv('allData.csv')


# In[8]:


#importing saved compiled data to avoid running combine_data
allData = pd.read_csv('allData.csv')
allData = allData.drop(columns = 'Unnamed: 0')
allData.describe()


# In[9]:


labworkData = allData[allData['Activity'] == 'Work In Lab'].reset_index(drop=True)
walkingData = allData[allData['Activity'] == 'Walking'].reset_index(drop=True)
runningData = allData[allData['Activity'] == 'Running'].reset_index(drop=True)
drivingData = allData[allData['Activity'] == 'Driving'].reset_index(drop=True)
relaxingData = allData[allData['Activity'] == 'Relaxing'].reset_index(drop=True)
eatingData = allData[allData['Activity'] == 'Eating'].reset_index(drop=True)
unlistedData = allData[allData['Activity'] == 'Not in List'].reset_index(drop=True)
activityDfList = [labworkData,walkingData,runningData,drivingData,relaxingData,eatingData,unlistedData]


# In[92]:


prunedDataList = []
for df in activityDfList:
    if df['Activity'].iloc[1] != 'Work In Lab' and df['Activity'].iloc[1] != 'Not in List':
        prunedDataList.append(remove_outliers_df(df))


# In[11]:


chunkedData = chunk(prunedDataList[3])


# In[12]:


chunkedData[0].describe()


# In[70]:


fftData = get_fft_values(chunkedData[0])
fftData.head()


# In[37]:


chunkedData[0]


# In[114]:


peak_detection(fftData, 'gX', 6)


# In[115]:


peak_detection(fftData, 'gZ', 12)


# In[ ]:
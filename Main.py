import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml

#import the raw data for accelerometer, gyroscope, and labels
accelData = pd.read_csv('C:\\Users\\MichaelK\\Documents\\SeniorDesign\\accelerometer_data.txt')
gyroData = pd.read_csv('C:\\Users\\MichaelK\\Documents\\SeniorDesign\\gyroscope_data.txt')
labelData = pd.read_csv('C:\\Users\\MichaelK\\Documents\\SeniorDesign\\activity_data.txt', keep_default_na=False)


#combines all data into one file, correlating accel and gyro time values and applying a label to every entry
#inputs - 3 dataframes containing info from accelerometer, gyroscope, and activity
#output - dataframe containing combined info with correlated datapoints

# !!! This function can take hours to run depending on the system being used !!!
#save the resulting dataframe to a file after running so it doesnt need to be used twice on the same data
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
            if labelTime < accelTime: #This works due to the format of the activity data file
                labelIndex = i
                currentLabel = labelData['Activity'][i]
                #According to Ali, NULL and unanswered can be interpreted as whatever the most recent label is
                if currentLabel == 'NULL' or currentLabel == 'Unanswered' or currentLabel == 'No Change':
                    currentLabel = oldLabel
            else:
                break

        #test print -- uncomment this to ensure the function is working if it takes too long

        if oldLabel != currentLabel:
            print('Current Label: ' + currentLabel)

        oldLabel = currentLabel #used when NULL or Unanswered is encountered

        #Saving the current gyroscope timestamp as a datetime object
        gyroTime = datetime.strptime(gyroData['Time'][gyroIndex], '%m/%d/%y %H:%M:%S.%f')


        if gyroIndex < gyroData['Time'].count()-1: #Preventing out of bounds
            nextGyroTime = datetime.strptime(gyroData['Time'][gyroIndex + 1], '%m/%d/%y %H:%M:%S.%f')
            while abs(accelTime-gyroTime) > abs(accelTime-nextGyroTime): #location in gyroscope data never needs to decrement
                gyroIndex += 1
                if gyroIndex == gyroData['Time'].count()-1: #if we get to the last datapoint in gyro
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
def chunk_data(df): #input dataframe
    chunkCount = len(df.index)//1000 #determining the amount of adjacent chunks that will fit
    chunkList = []
    if chunkCount == 1:
        chunk1 = df.iloc[0:1000]
        chunk2 = df.iloc[-1000:]
        chunkList.append(chunk1)
        chunkList.append(chunk2)
        return chunkList
    for i in range(chunkCount-1):
        tempChunk = df.iloc[i*1000:i*1000+1000] #normal chunks
        tempChunk.reset_index(drop = True, inplace = True)
        chunkList.append(tempChunk)
        if i*1000+1500 <= len(df.index):
            tempChunk = df.iloc[i*1000 + 500:i*1000+1500] #offset chunks]
            tempChunk.reset_index(drop = True, inplace = True)
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
#these values may be modified to improve FFT generation
t_n = 0.1
N = 1000
T = t_n / N
f_s = 1/T


#this function finds frequency from time for a chunk of data
#input - chunk (dataFrame)
#output - outData (dataFrame) containing frequency and fft values for all sensors and axis
#code from http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
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

#plots data against time for a column in a dataset
#inputs - df (dataFrame), col (String)
#output - plot
def plot_axis(df, col):
    #determining sensor and axis from col
    if col[0] == 'g': sensor = 'gyroscope '
    else: sensor = 'accelerometer '
    if col[1] == 'X': axis = 'X axis'
    elif col[1] == 'Y': axis = 'Y axis'
    else: axis = 'Z axis'

    #plot the amplitude angainst the frequency
    plt.plot(df['Frequency'], df[col], linestyle='-', color='blue')
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.title("Frequency domain of the " + sensor + axis, fontsize=16)
    plt.show()
    return

#runs auto correlation on a column of data
#inputs - col (list)
#outputs - correlation resutl
def autocorr(col):
    result = []
    result = np.correlate(col, col, mode='full')
    return result[len(result)//2:]


#uses autocorr function to produce array of correlation values
#derived from a function by Ahmet Taspinar
#http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
def get_autocorr_values(df):
    t_n = 0.1
    N = 1000
    T = t_n / N
    f_s = 1/T
    x_values = np.array([T * jj for jj in range(0, N)])
    outData = pd.DataFrame(columns = ['x_values','aX','aY','aZ','gX','gY','gZ'])
    outData['x_values'] = x_values
    for key, value in df.iteritems():
        if key != 'Time' and key != 'Activity':
            outData[key] = autocorr(df[key])

    return outData



#function produced as a modification of code written by basj on stack overflow
#https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
#inputs - df (dataFrame), col (String), cnt (Int) - cnt is the number of peaks to identify, plt (Boolean)
#output - plot
def peak_detection(df, col, cnt, dist, plot):
    #determining sensor and axis from col
    if col[0] == 'g': sensor = 'gyroscope '
    else: sensor = 'accelerometer '
    if col[1] == 'X': axis = 'X axis'
    elif col[1] == 'Y': axis = 'Y axis'
    else: axis = 'Z axis'
    x = df[col]
    prom = 10 #using prominence to find peaks
    peaks2, _ = find_peaks(x, prominence=prom, distance = dist)
    while len(peaks2) < cnt: #reducing prominence until correct amount of peaks is reached
        prom -= 0.00025
        peaks2, _ = find_peaks(x, prominence=prom, distance = dist)

    if plot: #if the user wants to plot the results
        plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['peaks'])
        plt.title("Peak Detection for the " + sensor + axis, fontsize=16)
        plt.show()
    return x[peaks2]


#splits a main dataFrame into subset based on the activity label
#input - df (dataFrame)
#output - outList (list) of (dataFrame)
def split_activities(df):
    outList = []
    activityList = []
    for activity in df['Activity']: #creating a list of activities
        exists = False
        for i in activityList:
            if activity == i:
                exists = True
        if not exists:
            activityList.append(activity)
    for i in activityList: #seperating main dataframe by activity
        outList.append(df[df['Activity'] == i].reset_index(drop=True))
    return outList


#Extracts the power spectral density from each column in a given chunk
#derived from a function by Ahmet Taspinar
#http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
def get_psd_values(chunk):
    t_n = 0.1
    N = 1000
    T = t_n / N
    f_s = 1/T
    f_values, psd_aX = welch(chunk['aX'], fs=f_s)
    f_values, psd_aY = welch(chunk['aY'], fs=f_s)
    f_values, psd_aZ = welch(chunk['aZ'], fs=f_s)
    f_values, psd_gX = welch(chunk['gX'], fs=f_s)
    f_values, psd_gY = welch(chunk['gY'], fs=f_s)
    f_values, psd_gZ = welch(chunk['gZ'], fs=f_s)
    outInfo = {'Frequency':f_values,'aX':psd_aX,'aY':psd_aY,'aZ':psd_aZ,'gX':psd_gX,'gY':psd_gY,'gZ':psd_gZ}
    outData = pd.DataFrame(outInfo)
    return outData

#finds change in acceleration for six sub chunks of a larger chunk
#inputs - col (list)
#outputs - col (list)
def jerk_col(col):

    jerkStart = [col[0],col[167],col[333],col[500],col[666],col[833]]
    jerkEnd = [col[168],col[334],col[501],col[667],col[834],col[999]]
    jerkOut = np.zeros(6)

    for i in range(6):
        jerkOut[i] = jerkStart[i]-jerkEnd[i]

    return jerkOut

#uses jerk_col function to find the jerk values for 6 sub chunks of each column in a chunk
#inputs - chunk (dataFrame)
#outputs - outData (dataFrame)
def find_jerk(chunk):
    outData = pd.DataFrame(columns = ['count','aX','aY','aZ','gX','gY','gZ'])
    outData['count'] = [1,2,3,4,5,6] #outputs 6 jerk values for each sensor axis
    for key, value in chunk.iteritems(): #itterating over the chunk's columns
        if key != 'Time' and key != 'Activity':
            outData[key] = jerk_col(value)

    return outData

#finds the amount of zero crossing points for a column of data
#inputs - col (list)
#outputs - timesCrossed (int)
def zero_cross_col(col):
    timesCrossed = [0];
    lastVal = 0
    for i in col:
        if i < 0 and lastVal > 0:
            timesCrossed[0] += 1
        elif i > 0 and lastVal < 0:
            timesCrossed[0] += 1
        lastVal = i
    return timesCrossed

#uses zero_cross_col to find zero crossing points for each column in a data chunk
#inputs - chunk (dataFrame)
#outputs - outData (dataFrame)
def find_zero_cross(chunk):
    outData = pd.DataFrame(columns = ['aX','aY','aZ','gX','gY','gZ']) #dataframe will only have depth of 1
    for key, value in chunk.iteritems(): #itterating over the chunk's columns
        if key != 'Time' and key != 'Activity':
            outData[key] = zero_cross_col(value)

    return outData

#fins mean of 6 subsections of a column of data
#inputs - col(array)
#outputs - meanOut (array) of 6 means
def mean_col(col):
    meanSum = np.zeros(6)
    meanCount = np.zeros(6)
    meanOut = np.zeros(6)
    i = 0
    while i < 1000:
        if i < 1000/6:
            meanSum[0] += col[i]
            meanCount[0] += 1
        if i < 2000/6:
            meanSum[1] += col[i]
            meanCount[1] += 1
        if i < 3000/6:
            meanSum[2] += col[i]
            meanCount[2] += 1
        if i < 4000/6:
            meanSum[3] += col[i]
            meanCount[3] += 1
        if i < 5000/6:
            meanSum[4] += col[i]
            meanCount[4] += 1
        if i < 6000/6:
            meanSum[5] += col[i]
            meanCount[5] += 1
        i += 1

    for j in range(6):
        meanOut[j] = meanSum[j]/meanCount[j]

    return meanOut

#Uses mean_col function to find the mean values for a data chunk
#inputs - chunk (dataFrame)
#outputs - outData (dataFrame)
def find_mean(chunk):
    outData = pd.DataFrame(columns = ['count','aX','aY','aZ','gX','gY','gZ'])
    outData['count'] = [1,2,3,4,5,6] #outputs 6 averages for each sensor axis
    for key, value in chunk.iteritems(): #itterating over the chunk's columns
        if key != 'Time' and key != 'Activity':
            outData[key] = mean_col(value)

    return outData


#this function takes a list of dataFrames divided by activity and extracts their features
#input - dfList (List) of (DataFrame)
#output - outData (DataFrame) with a row of features for each chunk
def extract_features(dfList):
    columnList = []
    for i in range(72):
        columnList.append('Feature ' + str(i))
    columnList.append('Activity')
    outData = pd.DataFrame(columns = columnList)

    for df in dfList:
        chunkList = chunk_data(df)
        #print(len(chunkList))
        i = 0
        for chunk in chunkList:
            #print(i)
            fftData = get_fft_values(chunk)
            psdData = get_psd_values(chunk)
            #corData = get_autocorr_values(chunk)
            featNum = 0
            outInfo = {}
            for key, value in fftData.iteritems():
                if key != 'Frequency' and key != 'Activity':
                    fftPks = peak_detection(fftData, key, 6, 1, False)[0:6]
                    psdPks = peak_detection(psdData, key, 6, 1, False)[0:6]
                    for j in range(6):
                        outInfo.update([('Feature '+ str(featNum),fftPks.iloc[j]),('Feature '+str(featNum+1),psdPks.iloc[j])])
                        #outInfo.update([('Feature '+ str(featNum),[fftPks.index[j],fftPks.iloc[j]]),('Feature '+str(featNum+1),[psdPks.index[j],psdPks.iloc[j]])])
                        featNum += 2
            i += 1
            outInfo.update([('Activity',chunk['Activity'].iloc[0])])
            outData = outData.append(outInfo,ignore_index = True)
    return outData

###############################################################################

try:
    #importing saved compiled data to avoid running combine_data
    allData = pd.read_csv('allData.csv')
    allData = allData.drop(columns = 'Unnamed: 0')
    allData.describe()

#only use once per input data set - correlates the data
except: #if there is not an allData file saved
    allData = combine_data(accelData,gyroData,labelData)
    allData.to_csv('allData.csv')


#seperates the main data by activity
activityDfList = split_activities(allData)

#removes outlying datapoints from each activity
prunedDataList = []
for df in activityDfList:
    if df['Activity'].iloc[1] != 'Work In Lab' and df['Activity'].iloc[1] != 'Not in List':
        prunedDataList.append(remove_outliers_df(df, 4))


#creating the model
clf = RandomForestClassifier(n_estimators=1000, max_depth=3,random_state=0)

#extracting features for training
trainingData = extract_features(prunedDataList[0:3])

#training the model
noLabelData = trainingData.drop(['Activity'],axis = 1)
clf.fit(noLabelData, trainingData['Activity'])

#saving the model to a file
model = pickle.dumps(clf)
modelFile = open('seniodDesignModel.txt','wb')
modelFile.write(model)

#import the test data for accelerometer, gyroscope, and labels
testAccelData = pd.read_csv('C:\\Users\\Micha\\Documents\\SeniorDesign\\testdata\\accelerometer_data (1).txt')
testGyroData = pd.read_csv('C:\\Users\\Micha\\Documents\\SeniorDesign\\testdata\\gyroscope_data (1).txt')
testLabelData = pd.read_csv('C:\\Users\\Micha\\Documents\\SeniorDesign\\testdata\\activity_data (1).txt', keep_default_na=False)

rawTestData = combine_data(testAccelData,testGyroData,testLabelData)
testDfList = split_activities(rawTestData)
#removes outlying datapoints from each activity
prunedTestData = []
for df in testDfList:
    if df['Activity'].iloc[1] != 'Work In Lab' and df['Activity'].iloc[1] != 'Not in List':
        prunedTestList.append(remove_outliers_df(df, 4))

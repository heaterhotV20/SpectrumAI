import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

"""date      , hh,min,ss      ,startfreq,end freq , ignore , ignore, 6:2006 (2001 values of freq)"""

###Step 1: Process Input file from csv with format above to produce Spectrogram images and store them in NumPy Arrays - reGenImage and reGenData

folder = 'rfData/DTTA_L22_2100H/'
for gainFile in os.listdir(folder):
	filePath = folder+gainFile
	destPath = 'reGenData/DTTA_L22_2100H/' +gainFile
	destImgPath = 'reGenImage/DTTA_L22_2100H/' +gainFile
	if(not os.path.isdir(destPath)):
			os.mkdir(destPath)
	if(not os.path.isdir(destImgPath)):
			os.mkdir(destImgPath)

	## We order the files in ascending order
	minList = []
	Filelist = os.listdir(filePath)
	for file in Filelist:
		minList.append(int(file.split('-')[0]))
	Filearray = np.array(Filelist)
	orderedFiles = Filearray[np.argsort(minList)].tolist()[::-1]
	print(orderedFiles)




	for file in orderedFiles:
		dataPath = os.path.join(filePath,file)
		print("###### Opening file:", dataPath)
		df = pd.read_csv(dataPath,
			index_col = None, 
			header = None, 
			sep =',')

		df = df.head(10000) # Get first ~500 time samples
		numRows = df.iloc[:,1].nunique()
		print("Number of Time samples:", numRows)

		minFreq = int(file.split('-')[0])
		maxFreq = int(file.split('-')[1].split('.')[0])

		spectral = np.empty((numRows,2001*(maxFreq-minFreq)//5))

		timeSeen = []
		timeCount = 0
		background = np.ones(2001)*-75

		

		destPic = os.path.join(destImgPath,file.split('.')[0] + '.png')
		destNp =  os.path.join(destPath,file.split('.')[0])

		#Reorder frequency of input data and fill empty values with that of background
		for time in df.iloc[:,1]:
			if time not in timeSeen:
				powerAtTime = []
				dfTime = df[(df.iloc[:,1] == time)]
				for freq in range(minFreq*1000000,maxFreq*1000000,5*1000000):
					dfFreq = dfTime[(dfTime.iloc[:,2] == freq)]
					if(dfFreq.empty):
						powerAtTime.extend(background.tolist())
					else:
						powerAtTime.extend(dfFreq.values[0,6:2007].tolist())
				
				spectral[timeCount,:] = np.array(powerAtTime).T
				timeCount = timeCount+1

				timeSeen.append(time)

		fig = plt.figure()
		plot = plt.pcolormesh(spectral,cmap = 'gnuplot',vmin = -82, vmax = -35)
		plt.colorbar()
		plt.savefig(destPic)
		plt.close(fig)
		
		np.save(destNp,spectral)
		del spectral
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Find start and end frequencies and then trim spectral images 

npfile = '500-600.npy'
textFile = 'reGenData/DTTA_L12_2000H/l32g10/' + npfile 
lookahead = 100
threshold =17
minFreq = int(npfile.split('-')[0])
maxFreq = int(npfile.split('-')[1].split('.')[0])


"""date      , hh,min,ss      ,startfreq,end freq , ignore , ignore, 6:2006 (2001 values of freq)"""
"""500-600.npy, 800-900.npy, 2600-2700.npy,  2100-2200.npy, 2300-2400.npy,  lookahead = 100, abs =17"""
"""2500-2600.npy, lookahead =100,abs =15,max"""


"""Didnt work well : 2500-2600.npy, 2400-2500"""

locateSpectral = np.load(textFile)
print("####Spectral search on: ", textFile, "with shape: ",locateSpectral.shape)

specMean = np.mean(locateSpectral,axis=0) #Average spectrum across time axis
diffSpec =specMean[lookahead:] - specMean[:-lookahead]
bins = diffSpec.shape[0]
prevIndx = 0
waitStart = True

listSignals =[]
listFreq = []

##Algo to find pairs of start,end frequencies
for num,i in enumerate(diffSpec):
	if(abs(i) > threshold):
		if(i>0 and (num  > prevIndx+lookahead-1) and waitStart):
			startFreq = (num+lookahead)/bins*(maxFreq-minFreq)+minFreq
			startIdx = num+lookahead
			#print("Start of Bandwidth",i,"Freq:",startFreq)
			prevIndx = num
			waitStart = False
			

		elif(i<0 and (num  > prevIndx+lookahead-1) and not waitStart):
			endFreq = (num+lookahead)/bins*(maxFreq-minFreq)+minFreq
			endIdx = num+lookahead
			#print("End of Bandwidth",i,"Freq:",endFreq)
			prevIndx = num
			waitStart = True
			
			if(endFreq-startFreq>5):
				listSignals.append((startIdx,endIdx))
				listFreq.append((int(startFreq),int(endFreq)))
				print("Signal identified between:", startFreq, "and" , endFreq )

#Once spectral bands found, trim and save the spectral image:
timeIntervals =25
saveDest ='newTrain/'

for timeDate in os.listdir('reGenData/'):
	timeDateDir = os.path.join('reGenData',timeDate)
	for gain in os.listdir(timeDateDir):
		gainDir = os.path.join(timeDateDir,gain)
		NUMPYFILE = os.path.join(gainDir,npfile)
		spectral = np.load(NUMPYFILE)
		print("Loaded:", NUMPYFILE)

		for signalRange in listSignals:
			startFreq = int(signalRange[0]/bins*(maxFreq-minFreq)+minFreq)
			endFreq = int(signalRange[1]/bins*(maxFreq-minFreq)+minFreq)
			folderDest = saveDest +str(startFreq) +'-' +str(endFreq) 
			if(not os.path.isdir(folderDest)):
				os.mkdir(folderDest)

			print("Trimming Spectrum Frequency:", startFreq, " to ", endFreq)

			midPoint = (signalRange[0] +signalRange[1])//2
			startBand, endBand = max(midPoint-4000,0), min(midPoint+4000,40019)
			for timeStart in range(0,spectral.shape[0]-timeIntervals,timeIntervals):
				print("time:",timeStart)
				fig = plt.figure()		
				plt.pcolormesh(spectral[timeStart:timeStart+timeIntervals,startBand:endBand],cmap = 'gnuplot',vmin = -82, vmax = -35)
				plt.axis('off')
				plt.savefig(os.path.join(folderDest,timeDate+'-'+gain+'_'+str(timeStart)+".png"))
				plt.close(fig)


	
# Import stuff
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import os,sys
import datetime
from itertools import repeat

# Set path variables
#globalPath = Path('/Users/datalab1/Lars/Cyclomedia_panoramas/') #Laptop
globalPath = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/') #DGX
downloadDir = globalPath / 'downloaded_panoramas'
downloadDir2 = globalPath / 'downloaded_panoramas_single'
dirfile = Path('/Users/datalab1/Lars/Panorama_Amsterdam/dir.txt')
dirfile = globalPath / 'list_of_Adam_figures.txt' 

# Set global url to download. lon+lat to be filled in in the function
baseurl = 'https://atlas.cyclomedia.com/PanoramaRendering/RenderByLocation2D/'
EPSG = '4326'
lon = '%s'
lat = '%s'
width = '?width=1024'
height = '&height=512'
hfov = '&hfov=90'
index = '&index=%s'
apiKey = '&apiKey=KXchCnQRqvsNmvgpa_Epy7stuX8dWrIVmvUd6309aPkJeXIpaTsRqHQVG_Ix6rsN'
url = baseurl+EPSG+'/'+lon+'/'+lat+width+height+hfov+index+apiKey 

# Get dataframe with all the coordinates
df = pd.read_csv(str(dirfile),delimiter=',',usecols=[0],names=['filename'])
df2 = df['filename'].str.split('_',expand=True)
df = df.join(df2)

df.columns = ['Filename','Rommel','Nummer','Longitude','Latitude']

# Actual download function
def download(lat,lon,Nummer,url,downloadDir):
    for index,name in enumerate(['_L','_C','_R']):
        r=requests.get(url %(str(lon),str(lat),index), auth=('rws_develop', 'mwlx8lpr'))
        assert r.status_code == 200
        with open(str(downloadDir / ('Cyclo_'+str(Nummer)+name+'_'+lon+'_'+lat+'.jpg')), 'wb') as f_out:
            r.raw.decode_content = True
            for chunk in r.iter_content(1024):
                f_out.write(chunk)
    

# get max number of cores
nr_of_cores = mp.cpu_count()

# let the multicore magic distribute the function to all the cores 
print('Busy with multicore part...')
a = datetime.datetime.now()
pool = mp.Pool(processes = 75) # zijn er nog 5 over voor andere dingen:P
pool.starmap(download,zip(df['Latitude'].values,df['Longitude'].values,df['Nummer'].values,repeat(url),repeat(downloadDir)))
b = datetime.datetime.now()
c = b - a
print('Multicore done in seconds: ', c.total_seconds())



#print('Busy with singlecore part...')
#a = datetime.datetime.now()
#for lat,lon,Nummer in zip(df['Latitude'].values,df['Longitude'].values,df['Nummer'].values):
#    download(lat,lon,Nummer,url,downloadDir2)
#b = datetime.datetime.now()
#c = b - a
#print('Singlecore in seconds: ', c.total_seconds())



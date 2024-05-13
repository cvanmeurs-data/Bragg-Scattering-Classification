#!/usr/bin/env python
# coding: utf-8

# In[4]:


import ee
 
# Trigger the authentication flow.
ee.Authenticate()


# In[7]:


# Initialize project
ee.Initialize(project='ee-cvanmeursdata-thesis')


# In[99]:


# Import packages. 
import os
import ee
import geemap
import argparse
import folium
from geemap import ml  # note new module within geemap
import pandas as pd
from sklearn import ensemble
from datetime import datetime
from datetime import date


# In[9]:


# Set output directory.
out_dir = os.path.expanduser("~/Downloads")

# Ensure output directory is set to downloads. 
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# In[10]:


# Define bands used in training and classification. 
trainBands = [
"VH_E_mean",
"VH_asm_mean",
"VH_contrast_mean",	
"VH_corr_mean",	
"VH_dent_mean",	
"VH_diss_mean",	
"VH_dvar_mean",	
"VH_ent_mean",
"VH_idm_mean",	
"VH_inertia_mean",	
"VH_mean",
"VH_prom_mean",	
"VH_savg_mean",	
"VH_sent_mean",	
"VH_shade_mean",	
"VH_svar_mean",	
"VH_var_mean",
"VV_E_mean",	
"VV_asm_mean",	
"VV_contrast_mean",	
"VV_corr_mean",	
"VV_dent_mean",	
"VV_diss_mean",
"VV_dvar_mean",	
"VV_ent_mean",	
"VV_idm_mean",
"VV_inertia_mean",
"VV_mean",
"VV_prom_mean",	
"VV_savg_mean",	
"VV_sent_mean",	
"VV_shade_mean",	
"VV_svar_mean",	
"VV_var_mean",	
"class"
]
feature_names = trainBands[:-1]

# Identify label used for classification. 
label = "class"

# Compile list of the selected Bands: VV, VH, Ent, GLCM.
selectedBands = ee.List([
  "VV",
  "VH",
  "VV_E",
  "VH_E",
  "VV_asm",
  "VV_contrast",
  "VV_corr",
  "VV_var",
  "VV_idm",
  "VV_savg",
  "VV_svar",
  "VV_sent",
  "VV_ent",
  "VV_dvar",
  "VV_dent",
  "VV_diss",
  "VV_inertia",
  "VV_shade",
  "VV_prom",
  "VH_asm",
  "VH_contrast",
  "VH_corr",
  "VH_var",
  "VH_idm",
  "VH_savg",
  "VH_svar",
  "VH_sent",
  "VH_ent",
  "VH_dvar",
  "VH_dent",
  "VH_diss",
  "VH_inertia",
  "VH_shade",
  "VH_prom",
]);


# In[11]:


# Import Training Data. 
trainingData = pd.read_csv('VV_VH_Ent_GLCM_TrainingData.csv')

# Import the selectedLakes and randomForest
randomForest = ee.FeatureCollection('projects/ee-cvanmeursdata-thesis/assets/RandomForest_2') 
selectedLakes = ee.FeatureCollection('projects/ee-cvanmeursdata/assets/Thesis/Selected_Lakes')

# Get the features and labels into separate variables.
X = trainingData[feature_names]   # Image bands. 
y = trainingData[label]           # class


# In[12]:


# Create a Random Forest classifier and fit. 

# Amount of Trees. 
n_trees = 20
# Type of Classifier. 
rf = ensemble.RandomForestClassifier(n_trees).fit(X, y)
# Compile trees to string. 
trees = ml.rf_to_strings(rf, feature_names)

# Create a ee classifier to use with ee objects from the trees
ee_classifier = ml.strings_to_classifier(trees)


# In[74]:


# Define the function necessary for outlining, clustering and classification.  

# afn_SNIC: Creates the SNIC clustering images. 
def afn_SNIC(imageOriginal, sPS):
#   Adjustable Superpixel Seed and SNIC segmentation Parameters:
    superPixelSize = sPS;
    compactness = 0.1;                            # Adjust
    connectivity = 4;                             # Adjust
    seedShape = 'square';                         # Adjust
    neighborhoodSize = 2 * superPixelSize;        # Adjust
  
    theSeeds = ee.Algorithms.Image.Segmentation.seedGrid(superPixelSize, seedShape);
    snic = ee.Algorithms.Image.Segmentation.SNIC(**{
        'image': imageOriginal,
        'size': superPixelSize,
        'compactness': compactness,
        'connectivity': connectivity,
        'neighborhoodSize': neighborhoodSize,
        'seeds': theSeeds
    });
    theStack = snic.addBands(theSeeds);
    return (theStack);


# vectorize: Creates the vectors from the SNIC clustering images. 
def vectorize(snicImage, geometry):
    vectors = ee.Image(snicImage).select('clusters').reduceToVectors(**{
        'geometryType': 'polygon', 
        'reducer': ee.Reducer.countEvery(), 
        'scale': 3,
        'maxPixels': 1e10, 
#     tileScale: 8, 
        'bestEffort': True,
        'geometry': geometry
  });
    return vectors;

# computeEntropy: Computes entropy for all relevant bands.  
def computeEntropy(image):
# Define a neighborhood with a kernel.
    kernel = ee.Kernel.circle(**{'radius': 4});
    ent =  ee.Image(image.toInt32()).entropy(kernel).rename('VV_E', 'VH_E');
    return image.addBands(ent);


# computeGlcm: Computes the GLCM values and add them as bands.  
def computeGlcm(image):
    glcm = ee.Image(image.toInt32()).glcmTexture(4); 
    return image.addBands(glcm);


# In[ ]:


def autoSizing(image):
    
#     Dates and bufferSize. 
    date = image.get('system:time_start') 
    bufferSize = AOI.centroid().buffer(300)
    
#     Compute lower and upper limit for dates used in outlining. 
    lowDate = ee.Date(date).advance(-2, 'month').format('YYYY-MM-dd') #.getInfo()
    highDate = ee.Date(date).advance(2, 'month').format('YYYY-MM-dd') #.getInfo()

#     Compute median image
    imageMedian = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(AOI).filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).filter(ee.Filter.date(lowDate, highDate)).select('VV', 'VH').median().clip(AOI.buffer(3000).bounds())
    
#     Compute the Sigma VHVV Image and add to the map. 
    vhvv = ee.Image(imageMedian.select('VV').multiply(imageMedian.select('VH'))).pow(0.5).rename('VHVV');
    
#     Compute dictionary for the median value within the buffer zone. 
    means = vhvv.reduceRegion(**{
      'reducer': ee.Reducer.median(),
      'geometry': bufferSize, 
      'scale': 50
    });   
    
#   Compute dictionary for the standard deviation value within the buffer zone. 
    stdDev = vhvv.reduceRegion(**{
      'reducer': ee.Reducer.stdDev(), 
      'geometry': bufferSize, 
      'scale': 50
    });
          
#     Compute the lower threshold for water/shore. 
    thresholdLow = ee.Number(means.get('VHVV')).subtract(ee.Number(5).multiply(ee.Number(stdDev.get('VHVV'))));
#     Compute the upper threshold for water/shore. 
    thresholdHigh = ee.Number(means.get('VHVV')).add(ee.Number(5).multiply(ee.Number(stdDev.get('VHVV'))));
          
#     A maximum error is taken into account during computations. 
    geomMaxError= 10;
    areaMaxError = 10;
    
#     Sort the areas by size. 
    def setSize(f):
        return f.set({'size': f.geometry(geomMaxError).area(areaMaxError)})
    
#     Compute the zones within/outside of the threshold values, taking into account the pre-defined error margins. 
    zones = ee.Image(vhvv.gt(thresholdLow));
    zones = zones.updateMask(zones).reduceToVectors(**{
      'scale': 50
    }).map(setSize);
          
#     Sort the zones from large to small and add take the largest zone. 
#     Consequently add it to the map. 
    largestZone = zones.sort('size', False).toList(zones.size()).get(0);
    
#     Take the geometry of the relevant zone. 
    classGeometry = ee.Feature(largestZone).geometry();
    
#     The image at the selected date and location is called. The Entropy and GLCM are computed
#     and the relevant bands are selected. 
    classImage = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(AOI).filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).filter(ee.Filter.date(date, ee.Date(date).advance(1, 'day').format('YYYY-MM-dd'))).select('VV', 'VH').map(computeEntropy).map(computeGlcm).select(selectedBands).first();
          
#     Compute SNIC by importing the 'afn_SNIC' function.
    snic = afn_SNIC(classImage, 20);
          
#     Compute the vector outlines of the clusters by importing the 'vectorize' function. 
    vector = vectorize(snic, classGeometry);  
          
#     Classify the SNIC image, clipped to the geometry found in the autoSizing function.  
    classified = ee.Image(snic.select(feature_names)).clip(classGeometry).classify(ee_classifier);
    areaImage = ee.Image.pixelArea();
  
    finalImage = areaImage.addBands(classified);
  
    areas = finalImage.reduceRegion(**{
            'reducer': ee.Reducer.sum().group(**{
            'groupField': 1,
            'groupName': 'label'}),
            'geometry': classGeometry,
            'scale': 100,
            'maxPixels': 1e10}); 

    return ee.Feature(None, areas)
    


# In[149]:


# Basic Map setup. 
Map = geemap.Map()
Map.centerObject(lakeTono, 13);
Map.addLayer(lakeTono, {}, 'Lake Outline');

# The uploaded csv File with lats, lons, start and end dates. 
data = pd.read_csv('datesList.csv')
lats = data['lat'].values
lons = data['lon'].values
startDates = data['startDate'].values
endDates = data['endDate'].values


# Function: Input lon (number), lat (number), startDate (string) and endDate (string).
# Outputs an imageCollection for the requested values. 
def runTru(lon, lat, startDate, endDate):
    AOI = ee.Geometry.Point(lon, lat)
    image = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(AOI).filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).filter(ee.Filter.date(startDate1, endDate1)).select('VV', 'VH')
    return image



# In[156]:


# Run for all entries in the .csv file. 
# Run the autoSizing output and export the sums in a .csv file. 
for i in range(0, len(lats), 1):
    imageCol = runTru(lons[i], lats[i], startDates[i], endDates[i])
    autoResult = imageCol.map(autoSizing)
    output = ee.FeatureCollection(autoResult)
    geemap.ee_to_csv(output, filename='areas'+str(i)+'.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:





# In[ ]:





#import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

df_vnf = pd.read_csv('data/vnf_measurements_5pc_sample.csv')
#gdf = gpd.read_file('data/flare_clusters.geojson')
scatter_data = pd.read_csv('data/flare_clusters_scatter_data.csv')
record_counts = pd.read_csv('data/record_counts.csv')
facility_match_summary = pd.read_csv('data/facility_match_summary.csv')

"""
# Identifying gas flares from satellite data

This technical task is aimed at identifying flaring locations using satellite measurements, and tie them to oil and gas locations, so that we can find out which installations are flaring a lot.

I was given three datasets for the task; one csv file of VNF observations from satellite data and two shapefiles with locations of oil and gas facilities - one point and one polygon.

The VNF dataset is relatively large - over 10 million rows. The facilties shapefiles are smaller, around 17k locations across two files, but the polygon data required to represent their geometry results in some fairly unwieldy data sizes. For the purposes of this investigation I took a 5% sample of this dataframe to speed up exploration. The script I used to prepare the data and charts in this write up are available in this github repository.
"""
st.table(record_counts)

"""
The key task here was to investigate how to identify oil and gas installations and flares from a dataset that includes observations from a number of sources. The problem statement set out that the VNF observations included flare and non-flare observations but also that not all flaring observation would be near one of the oil and gas installation locations in the provided shapefiles.

Secondly, this is not an exhaustive dataset of all oil and gas installations in the world. My first thought was that this could be done with a classification algorithm such as logistic regressions, random forest or boosted trees. However, the lack of a well labelled set of data where each point can be robustly labelled as a flare or not means that this isn't possible using just the datasets available. 

Instead I did some Googling to understand a bit more about VNF data and how it has been used to identify flares in the past and found two useful heuristics that at least would allow me to identify flaring candidates and then piut forward some suggestions for developing this further.

## Defining heuristics

The two key pieces of information I used were that gas flares tend to be relatively hot compared to other observations in this satellite dataset and second, that whilst observations that come from biomass sources such as wildfires are often quite constrained in time, flares tend to occur across a longer period of time throughout the operation of a facility.

### Observation temperatures

The research suggests that flaring sites tend to burn at 1450 Kelvin and above. Plotting a histogram of the 5% sample of data I took for speed of analysis shows that there are three distinct peaks in the distribution the temperature of the observations and a distinct trough around the 1450K mark. Setting a threshold at 1450K and taking only observations above that threshold is a first start.
"""

st.plotly_chart(px.histogram(df_vnf, x='temp_bb'))

"""
### Clusters of observations and breadth of time window

In order to cluster observations quickly, I put a 500m buffer around each observation and then performed a unary union on these individual 500m buffer circles to create larger polygons where two or more observations were less than 500m apart. 

For each of these larger polygons I could then do a spatial join back to the original observation data to create summary stats for each larger cluster polygon including a total observation count, the min/max observation date and the range between the two, and the mean/median of the observation temperatures. 

I could then identify larger clusters where a) The median temperature was above 1450K and b) the observation range days was more than just a few days. Plotting average temperature against the number of observations, we can see that in the observations with a higher average temperature, there are more clusters with 25+ observations compared to those in the lower temperture group below 1450K.
"""

fig = px.scatter(scatter_data[scatter_data['observation_count'] > 3], 
                y='observation_count', 
                x='temp_bb_mean', 
                opacity=0.2, 
                width=700, 
                height=600, 
                color_continuous_scale=px.colors.sequential.Reds, 
                title='Average temperature against observation count for each observation cluster')

fig.update_traces(marker={'size': 4})
fig.update_yaxes(range=[0,200])
st.plotly_chart(fig)

"""
I also performed a spatial join with the facility polygons to identify observations that intersected with the locations in the facility dataset. Interestingly it seems like the 1450K threshold idea didn't hold up that well here. Only about 15% of the observation clusters that intersected with a facility had a median temperature over over 1450K.
"""
st.table(facility_match_summary)

"""
## Tagging facilties and observations

Although the spatial joins suggest the threshold method doesn't work that well, eyeballing these datasets on a map does give some interesting initial insights as to where the hot flaring sites are. For example these screenshots show parts of the North Sea where we can obviously see the facilities (in grey) that are flaring a lot (red blobs).
"""
north_sea = Image.open('img/north_sea.png')
st.image(north_sea, caption='Flare (red) and facility (grey) locations in the North Sea')

"""
We can also see the same for the area at the northern end of the Persian Gulf where we see quite large strings of sizable clusters of flaring activity.
"""

middle_east = Image.open('img/middle_east.png')
st.image(middle_east, caption='Flare (red) and facility (grey) locations in Persian Gulf area')

"""
## Next steps

This initial exploration only really scratched the surface of what could be done with this dataset. 

Using some simple heuristics to do with the frequency and temperature of observations does get oyu surprisingly far in identifying potential oil and gas flaring sites but more investigation would need to be done, probably with some more computing power as well to be able to run analyses on the full dataset.

It would be useful to create a properly labelled, robust dataset of known, verified observations of both oil and gas flaring and others so that this could be used as a dataset to train some classification algorithms.

It would also be useful to build this dataset into a proper geo-database that could be used in a genuinely interactive manner to spot check and analyse certain geogrpahical areas.
"""
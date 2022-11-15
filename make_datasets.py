import geopandas as gpd
import numpy as np
import pandas as pd
import shapely as sp


def load_gdf_and_set_crs(infile):
    gdf = gpd.read_file(infile)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf.to_crs(epsg=3857, inplace=True)
    
    return gdf


def make_facilities_gdf(point_file, polygon_file):
    print('Making facility dataset')
    # Load the point and area facility files, buffer the points and concatenate them into a single geodataframe

    facility_points = load_gdf_and_set_crs(point_file)
    facility_points.geometry = facility_points.geometry.buffer(500)

    facility_polyg = load_gdf_and_set_crs(polygon_file)

    facilities = pd.concat([facility_points, facility_polyg])
    facilities.to_file('data/facilities_combined.geojson')

    print('Facility dataset complete')
    return facilities


def make_vnf_sample(infile):
    print('Making vnf data sample')
    df = pd.read_csv('data/vnf_measurements_2017.csv')
    df = df.dropna()
    df.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    df = df.sample(frac=0.05)
    df.to_csv('data/vnf_measurements_5pc_sample.csv')
    print('vnf data sample complete')

    return df


def make_vnf_gdf(df):
    print('Making vnf geodataframe')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_gmtco, df.lat_gmtco))
    
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    
    gdf.to_crs(epsg=3857, inplace=True)

    # Set column types
    float_cols = ['lat_gmtco','lon_gmtco', 'temp_bb', 'radiant_heat_intensity', 'radiant_heat',
        'area_bb', 'sample_m10']

    for col in float_cols:
        gdf[col] = gdf[col].astype('float64')

    gdf['date_mscan'] = pd.to_datetime(vnf['date_mscan'])

    print('vnf geodataframe complete')
    return gdf


def make_flare_clusters(gdf):
    print('Making inital flare clusters')
    # Buffer flare observations and unary_union overlapping buffers as a crude method of clustering groups of flares

    clusters = gdf.copy()
    clusters.geometry = clusters.buffer(500)
    clusters = gpd.GeoDataFrame(geometry=gpd.GeoSeries([polygon for polygon in clusters.unary_union]))
    clusters.set_crs(epsg=3857, inplace=True)
    clusters['saved_geom'] = clusters.geometry

    print('Inital flare clusters complete')

    return clusters

def rename_tuple_columns(col):
    if isinstance(col, tuple):
        col = '_'.join(str(c) for c in col)
    return col

def make_flare_cluster_stats(clusters, vnf_data):
    print('Making flare cluster stats')

    vnf_cluster_join = vnf_data.sjoin(clusters, how='left')
    vnf_cluster_join.geometry = vnf_cluster_join['saved_geom']

    dissolved = vnf_cluster_join.dissolve(by='index_right', aggfunc={
            "id": "count",
            "date_mscan": ["min", "max"],
            "temp_bb":["mean", "median", "max"],
            "saved_geom":"first"       
            })

    dissolved.columns = map(rename_tuple_columns, dissolved.columns)
    dissolved.rename(columns={'id_count':'observation_count'}, inplace=True)
    dissolved['temp_over_1450'] = dissolved['temp_bb_mean'].apply(lambda x: True if x > 1450 else False)
    dissolved['observation_range_days'] = (dissolved['date_mscan_max'] - dissolved['date_mscan_min']).dt.days + 1
    dissolved.index.rename('index', inplace=True)

    facility_overlaps = dissolved.sjoin(facilities, how='left').groupby('index_right')['observation_count'].count()
    dissolved['facility_overlaps'] = facility_overlaps
    dissolved['facility_overlaps'] = dissolved['facility_overlaps'].fillna(0)
    dissolved['facility_match'] = dissolved['facility_overlaps'].apply(lambda x: True if x > 0 else False)
    dissolved[[col for col in dissolved.columns if col != 'saved_geom_first']].to_file('data/flare_clusters.geojson')

    print('Flare cluster stats complete')
    return dissolved

def make_facility_match_summary(df):
    (df.groupby(['facility_match', 'temp_over_1450'])['observation_count']
        .count()
        .reset_index()
        .to_csv('data/facility_match_summary.csv', index=False))

def make_flare_scatter_data(flare_clusters):
    flare_clusters[['observation_count', 'temp_bb_mean']].to_csv('data/flare_clusters_scatter_data.csv', index=False)

if __name__ == '__main__':
    print('Starting dataset build...')
    vnf = make_vnf_sample('data/vnf_measurements_2017.csv')
    vnf = make_vnf_gdf(vnf)

    facilities_point_shp = 'data/osm_oil_and_gas_points/osm_oil_and_gas_points.shp'
    faciltiies_polygon_shp = 'data/osm_oil_and_gas_polygons/osm_oil_and_gas_polygons.shp'
    facilities = make_facilities_gdf(facilities_point_shp, faciltiies_polygon_shp)

    flare_clusters = make_flare_clusters(vnf)
    flare_clusters = make_flare_cluster_stats(flare_clusters, vnf)

    make_facility_match_summary(flare_clusters)
    make_flare_scatter_data(flare_clusters)

    print('Dataset build complete!')
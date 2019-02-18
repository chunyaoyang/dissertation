# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:44:33 2019

@author: cyyang
"""
import os
import pandas as pd
import numpy as np



def findExcelFiles():
    """Load all excel files in folders"""
    pathes = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".xlsx"):
                if file.startswith("Sediment"):
                    pathes.append(root + "/" + file)
    return pathes

def read_sediment_measurement(f_path):
    raw_df = pd.read_excel(f_path,sheet_name='sediment measurement',skiprows=4,usecols=[5,11,20,21,23,26])
    raw_df.columns = ['Date', 'Q', 'C', 'Qs', 'Qt', 'ds']
    raw_df.Date = raw_df.Date.astype(str)
    raw_df.Date = pd.to_datetime(raw_df.Date.str[:10])
    return raw_df


def read_dischargeTimeSeries(f_path):       
    df_raw = pd.read_excel(f_path, sheet_name='stage-discharge(daily)',skiprows=3,thousands=',')
    df_raw = df_raw.drop(df_raw.columns[[0,2,5,8,11,14,17,20,23,26,29]], axis=1)   
    df = df_raw.apply(pd.to_numeric, errors='coerce')
       
    df2 = df[df.columns[1]].append(df[df.columns[3]]).append(df[df.columns[5]]).append(df[df.columns[7]]).append(df[df.columns[9]]).append(df[df.columns[11]]).append(df[df.columns[13]]).append(df[df.columns[15]]).append(df[df.columns[17]]).append(df[df.columns[19]]).reset_index(drop=True)
    
    Q = np.array(df2)

    df1 = df_raw[df_raw.columns[0]].append(df_raw[df_raw.columns[2]]).append(df_raw[df_raw.columns[4]]).append(df_raw[df_raw.columns[6]]).append(df_raw[df_raw.columns[8]]).append(df_raw[df_raw.columns[10]]).append(df_raw[df_raw.columns[12]]).append(df_raw[df_raw.columns[14]]).append(df_raw[df_raw.columns[16]]).append(df_raw[df_raw.columns[18]]).reset_index(drop=True)
    date = np.array(df1)
    
    df = pd.DataFrame()
    df['date'] = date
    df['Q'] = Q
    df = df.dropna(subset=['date'])
    return df



def load_attribute():
    elev = pd.read_csv('./data/elev.csv', header=0, usecols=[2,9,10,11], names=['Name', 'Elev',"MAX_Elev", 'Precip'])
    df1 = pd.read_excel('./data/Sediment Yield Field Data set_1 (Han River Watershed)/Data_set-Han_18May2016.xlsx',
                        sheet_name='Han R.(H1~H4)',header=0,usecols=[3,4,5,6])
    df2 = pd.read_excel('./data/Sediment Yield Field Data set_1 (Han River Watershed)/Data_set-Han_18May2016.xlsx',
                        sheet_name='Han R.(H5~H7)',header=0,usecols=[3,4,5])
    df3 = pd.read_excel('./data/Sediment Yield Field Data set_2 (Nakdong River Watershed)/Data_set-Nakdong_18May2016.xlsx',
                        sheet_name='Nakdong R.(N1~N5)',header=0,usecols=[3,4,5,6,7])
    df4 = pd.read_excel('./data/Sediment Yield Field Data set_2 (Nakdong River Watershed)/Data_set-Nakdong_18May2016.xlsx',
                        sheet_name='Nakdong R.(N6~N10)',header=0,usecols=[3,4,5,6,7])
    df5 = pd.read_excel('./data/Sediment Yield Field Data set_2 (Nakdong River Watershed)/Data_set-Nakdong_18May2016.xlsx',
                        sheet_name='Nakdong R.(N11~N14)',header=0,usecols=[3,4,5,6])
    df6 = pd.read_excel('./data/Sediment Yield Field Data set_5 (Seomjin River Watershed)/Data_set-Geum~Seomjin_18May2016.xlsx',
                        sheet_name='Geum R.(G1~G5)',header=0,usecols=[3,4,5,6,7])
    df7 = pd.read_excel('./data/Sediment Yield Field Data set_5 (Seomjin River Watershed)/Data_set-Geum~Seomjin_18May2016.xlsx',
                        sheet_name='Yeongsan R.(Y1~Y5)',header=0,usecols=[3,4,5,6,7])
    df8 = pd.read_excel('./data/Sediment Yield Field Data set_5 (Seomjin River Watershed)/Data_set-Geum~Seomjin_18May2016.xlsx',
                        sheet_name='Seomjin R.(S1~S4)',header=0,usecols=[3,4,5,6])

    dataList = [df1,df2,df3,df4,df5,df6,df7,df8]

    def removeEmpty(dataframe):
        df = np.array(dataframe)[3:35]
        df = np.delete(df, [2,24], axis=0)
        return df

    def bedsize(dataframe):
        df = dataframe.apply(pd.to_numeric, errors='coerce')
        df = np.array(df)[35:43]

        dmin = np.nanmin(np.float64(df),axis=0)
        dmax = np.nanmax(np.float64(df),axis=0)
        dmean = np.nanmean(np.float64(df),axis=0)
        return np.vstack((dmin,dmax,dmean))
    

    id_list = []
    for l in dataList:
        for n in l.columns:
            id_list.append(n)

    processedDataList = []
    for d in dataList:
        att = removeEmpty(d)
        bed = bedsize(d)
        processed_d = np.concatenate((att,bed), axis=0)
        processedDataList.append(processed_d)

    attri_data = np.hstack((processedDataList[0],processedDataList[1],processedDataList[2],processedDataList[3],
                       processedDataList[4],processedDataList[5],processedDataList[6],processedDataList[7]))
    attri_data = np.transpose(attri_data)


    attribute = pd.concat([pd.DataFrame(id_list,columns=['Name']),pd.DataFrame(np.float64(attri_data))],axis=1)
    attribute.rename(columns = {0:'Area'}, inplace = True)

    attribute = pd.merge(attribute, elev, on='Name')

    col_name = ['Name', 'lon', 'lat','Area', 'Avg_slope', 'Perimeter', 'Main_length',
                'Tributary_length', 'Total_length', 'Density', 'Width',
                'Slope_at_station', 'clay0', 'silt0', 'sand0', 'clay10',
                'silt10','sand10', 'clay30','silt30','sand30','clay50',
                'silt50','sand50','Urban','Agriculture','Forest',
                'Pasture','Wetland','Bare_land','Water','D_min', 'D_max', 'D_mean', 'Elev',"Max_Elev", 'Precip']
    attribute.columns = col_name
    attribute = attribute.set_index(["Name"])
    return attribute


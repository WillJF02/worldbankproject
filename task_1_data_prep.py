# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:56:21 2025

@author: wf2005
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import seaborn as sb
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Country(object):
    """
    Attributes
    ----------
    self.name : str
        contains the name of the country. This is so that we can
        search the list of Country objects using a given name and return the coresponding object
    
    self.df : dataframe
        contains the set of time series data for that country from the big dataset
    
    self.windows : list
        contains a list of dataframe subsets from self.df. Each dataframe
        subset contains a 5 year timespan. Since each country has a time series
        of size 44, this means there are a total of 40 windows per country.
    """
    
    
    def __init__(self,name, df):
        self.name = name
        self.df = df
        self.windows = [0]*40
        self.window_single = [0]
        
    def get_name(self):
        return self.name
    
    def get_df(self):
        return self.df
    
    def get_windows(self):
        return self.windows
    
    def get_last_window(self):
        return self.window_single

class CountryList(object):
    """
    Attributes
    -----------
    self.list : list
        a list that contains Country objects.
    """
    def __init__(self):
        self.list = []
        
    def get_country_data(self,name):
        """
        Parameters
        ----------
        name : STR
            Name of a country.

        Returns
        -------
        Country object
            returns the country object that has the corresponding id given.

        """
        
        return next((country for country in self.list if country.get_name() == name), None)
    
    def load_dataset(self,dataframe):
        """

        Parameters
        ----------
        dataframe : dataframe
            input the big dataset to seperate by country.

        Returns
        -------
        None.

        """  
        
        countries = dataframe.groupby("country")
        for country_name, country_df in countries:
            self.list.append(Country(country_name,country_df))
    
    def get_list(self):
        """

        Parameters
        ----------
        None.
        

        Returns
        -------
        List of Country Objects

        """
        return self.list
    
    def fill_gaps_impute(self):
        """

        Returns
        -------
        None.
        
        fills in missing gaps per country. For columns with no entries,
        the column is assigned with zeros.

        """
        imputer = SimpleImputer(strategy = "median")
        for country in self.list:
            numeric_cols = country.get_df().select_dtypes(include=["float64"]).columns
            impute_columns1 = country.get_df()[numeric_cols].columns[country.get_df()[numeric_cols].isnull().mean() <= 0.5]
            impute_columns2 = country.get_df()[numeric_cols].columns[country.get_df()[numeric_cols].isnull().mean() >= 0.01]
            impute_columns = impute_columns1.intersection(impute_columns2)
            if len(impute_columns) > 0:
                imputed_data = imputer.fit_transform(country.get_df()[impute_columns])  
                country.get_df()[impute_columns] = imputed_data
                
    def fill_gaps_arima_old(self):
        """

        Returns
        -------
        None.
        
        fills in missing gaps per country. For columns with no entries,
        the column is assigned with zeros.

        """
        for country in self.list:
            numeric_cols = country.get_df().select_dtypes(include=["float64"]).columns
            impute_columns1 = country.get_df()[numeric_cols].columns[country.get_df()[numeric_cols].isnull().mean() <= 0.5]
            impute_columns2 = country.get_df()[numeric_cols].columns[country.get_df()[numeric_cols].isnull().mean() >= 0.01]
            impute_columns = impute_columns1.intersection(impute_columns2)
            for column in impute_columns:
                series = country.get_df()[column].dropna()
                model = ARIMA(series, order=(1, 1, 1))
                model_fit = model.fit()
                missing_index = country.get_df()[column][country.get_df()[column].isnull()].index
                predictions = model_fit.predict(start=missing_index[0], end=missing_index[-1])
                country.get_df().loc[missing_index, column] = predictions
                
                
    def impute_gaps_arima(self):
        """

        Returns
        -------
        None.
        
        fills in missing gaps per country. For columns with no entries,
        the column is assigned with zeros.

        """
        for item in self.list:
            country = item.get_df().set_index("date")
            country.index = pd.to_datetime(country.index)
            #country.asfreq('AS-JAN')
            country = country.sort_index(ascending=True)
            test = item.get_df()
            numeric_cols = country.select_dtypes(include=["float64"]).columns
            impute_columns1 = country[numeric_cols].columns[country[numeric_cols].isnull().mean() <= 0.99]
            impute_columns2 = country[numeric_cols].columns[country[numeric_cols].isnull().mean() >= 0.01]
            impute_columns = impute_columns1.intersection(impute_columns2)
            for column in impute_columns:
                #series = country[column].dropna()
                series = country[column]
                model = ARIMA(series, order=(1, 1, 1))
                model_fit = model.fit()
                missing_index = series[series.isna()].index
                start_index = missing_index[0]
                end_index = missing_index[-1]
                predictions = model_fit.predict(start=start_index, end=end_index)
                predictions.replace(0,np.nan, inplace = True)
                country.loc[missing_index, column] = predictions
                
                
                if country[column].isna().any():
                    reverse_series = series.values[::-1]
                    reverse_series = pd.Series(reverse_series, index=series.index)
                    model = ARIMA(reverse_series, order=(1, 1, 1))
                    model_fit = model.fit()
                    missing_index = reverse_series[reverse_series.isna()].index
                    start_index = missing_index[0]
                    end_index = missing_index[-1]
                    predictions = model_fit.predict(start=start_index, end=end_index)
                    predictions.replace(0,np.nan, inplace = True)
                    predictions = predictions.sort_index(ascending=False)
                    
                    missing_index = series[series.isna()].index
                    country.loc[missing_index, column] = predictions.values
            
 
            country = country.sort_index(ascending=False)
            country = country.reset_index()
            country = country[['country'] + [col for col in country.columns if col != 'country']]
            item.df = country
             
    def convert_to_dataset(self):
        """
        Returns
        -------
        Dataframe
            Converts the list of country objects back into a singular dataframe

        """
        return pd.concat([country.get_df() for country in self.list], ignore_index = True)
    
    def create_sliding_windows(self):
        """
        Returns
        -------
        None.
        
        DESCRIPTION: Creates a sliding window on each country's time series data
        of 5 years. Creates a total of 40 windows each with 50 values in the dataframe

        """
        for country in self.list:
            country.windows = [0]*40
            #numeric_columns = country.get_df().select_dtypes(include=["float64"]).columns #selects the feature column indexes
            
            for i in range(0,len(country.get_df())-4):
                window = country.get_df().iloc[i:i+5]
                country.get_windows()[i] = window
                
    def create_sliding_windows_ten(self):
        """
        Returns
        -------
        None.
        
        DESCRIPTION: Creates a sliding window on each country's time series data
        of 5 years. Creates a total of 40 windows each with 50 values in the dataframe

        """
        
        for country in self.list:
            country.windows = [0]*35
            #numeric_columns = country.get_df().select_dtypes(include=["float64"]).columns #selects the feature column indexes
            
            for i in range(0,len(country.get_df())-9):
                window = country.get_df().iloc[i:i+10]
                country.get_windows()[i] = window         
            
    
    def clear_dataset(self):
        self.list.clear()
        
        
def create_impute():
    """create imputed data csv file"""

    """"imputed data initalisation""" 
    data = pd.read_csv("world_bank_data_dev.csv") #this variable will be altered
    data["date"] = pd.to_datetime(data["date"]) #changes the date values into datetime
    countries = CountryList() #creates the CountryList class
    countries.load_dataset(data) #seperates the dataset by country, adds the country's data into a dataframe as a class attribute                            

    """missing value impute"""
    countries.impute_gaps_arima() #fills missing data for each country
    data = countries.convert_to_dataset() #now missing values are filled in, turn it back into a big dataset for scaling

    #print(data.columns)
    """missing value fill"""
    #any remaining rows are filled in by the column average

    numeric_columns = data.select_dtypes(include=["float64"]).columns
    for column in numeric_columns:
        avg = original_data[column].median()
        data[column]=data[column].fillna(avg)
        
    """scaling dataset
    numeric_columns = data.select_dtypes(include=["float64"]).columns #selects the feature columns
    scaler = MinMaxScaler() #initalize scaler
    data[numeric_columns] = scaler.fit_transform(data.select_dtypes(include=["float64"])) #scales the data
    """

    
    countries.clear_dataset() #clears the preexisting list of countries so we can replace it with the scaled one
    countries.load_dataset(data) #adds the new scaled datset to become a list of Country objects
    countries.create_sliding_windows() #creates the sliding window for each country

    """
    exporting to csv file
    so that the programm doesn't have to run every time
    """
    data = countries.convert_to_dataset()
    data["date"]=data["date"].astype(str)
    data.to_csv("world_bank_data_imputed.csv", index=False)
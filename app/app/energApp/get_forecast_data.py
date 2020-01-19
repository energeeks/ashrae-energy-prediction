# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:04:59 2020

@author: Mufasa
"""
#%pip install requests
import requests
import configparser
import os
import pandas as pd
os.getcwd()
os.chdir("C:\\Users\\Mufasa\\Desktop\\Uni\\Master Biostatistik\\Git\\ashrae-energy-prediction\\app\\app\\EnergApp")

api_key = "54e35f48e2a81513b516113a868a6d7c"
lat = "48"
lon = "11"
call_beg = "https://api.openweathermap.org/data/2.5/forecast?lat="
call_mid = "&lon="
call_end = "&appid="
call = [call_beg, lat_value, call_mid, lon_value, call_end, api_key]
seperator = ""
seperator.join(call)

response = requests.get(seperator.join(call))
response.status_code == 200

response = requests.get("https://api.openweathermap.org/data/2.5/forecast?lat=",
                        lat_value,"&lon=",
                        lon_value, "&appid=54e35f48e2a81513b516113a868a6d7c")
response = requests.get("http://api.openweathermap.org/data/2.5/forecast?lat=35&lon=139&appid=54e35f48e2a81513b516113a868a6d7c")

print(response.json())


def get_api_key():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['openweathermap']['api_key']

api_key = get_api_key()

def get_forecast(api_key, lat, lon):
    url = "http://api.openweathermap.org/data/2.5/forecast?&units=metric&lat={}&lon={}&appid={}".format(lat, lon, api_key)
    r = requests.get(url)
    return r.json()

a = get_forecast(api_key, lat, lon)
b = a['list'][0]['main']
pd.DataFrame.from_records(b, index = [0])






if __name__ == '__main__':
    main()

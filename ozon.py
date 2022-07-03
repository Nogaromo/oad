#!/usr/bin/env python3

#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('longitude', metavar='LON', type=float, help='Longitude, deg')
#parser.add_argument('latitude',  metavar='LAT', type=float, help='Latitude, deg')

#if __name__ == "__main__":
    #args = parser.parse_args()
    #print(args.longitude, args.latitude)
#вдруг понадобится зачем-то   
    
 
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import json
import argparse
from geopy.geocoders import Nominatim

parser = argparse.ArgumentParser()
parser.add_argument('adress',  metavar='adrs', help='place somewhere on Earth', nargs='+')

if __name__ == "__main__":
    args = parser.parse_args()
    adress1 = args.adress
    if len(adress1) == 2:
        longitude = float(adress1[1])
        latitude = float(adress1[0])
    else:
        adress1[0] = Nominatim(user_agent='lol_kek_cheburek').geocode(adress1)
        longitude = adress1[0].longitude
        latitude = adress1[0].latitude
    print(longitude, latitude)

    file_path = '/home/nogaromo/Загрузки/MSR-2.nc'
    nc = netCDF4.Dataset(file_path)
    new_time = nc.variables['time'][:]-108
    for i in range(len(nc.variables['latitude'])):
        if abs(latitude - nc.variables['latitude'][i]) < 0.25:
            k_1 = i
    for i in range(len(nc.variables['longitude'])):
        if abs(longitude - nc.variables['longitude'][i]) < 0.25:
            k_2 = i


    jan = []
    for i in range(0, len(new_time), 12):
        jan.append(nc.variables['Average_O3_column'][i, k_1, k_2])
    jan = np.array(jan)
    jul = []
    for i in range(6, len(new_time), 12):
        jul.append(nc.variables['Average_O3_column'][i, k_1, k_2])
    jul = np.array(jul)
    #plotting data
    plt.subplots(figsize=(25.00, 10.00))
    plt.plot(nc.variables['time'][:], nc.variables['Average_O3_column'][:, k_1, k_2], label="All time", color='green')
    plt.plot(nc.variables['time'][0::12], jan, label="January", color='blue')
    plt.plot(nc.variables['time'][6::12], jul, label="July", color='red')
    plt.xlabel('Years')
    plt.ylabel('O$_3$ distribution in Balashikha')
    plt.grid()
    plt.xticks(nc.variables['time'][0::12], range(1978, 2020, 1), rotation=60)
    plt.legend()
    plt.title('Ozon distribution')
    plt.savefig('/home/nogaromo/Загрузки/ozon.png', dpi=400)
    data = {
      "coordinates": [latitude, longitude],
      "jan": {
        "min": float(jan.min()),
        "max": float(jan.max()),
        "mean": float(jan.mean().round(1))
      },
      "jul": {
        "min": float(jul.min()),
        "max": float(jul.max()),
        "mean": float(jul.mean().round(1))
      },
      "all": {
        "min": float(nc.variables['Average_O3_column'][:, k_1, k_2].min()),
        "max": float(nc.variables['Average_O3_column'][:, k_1, k_2].max()),
        "mean": float(nc.variables['Average_O3_column'][:, k_1, k_2].mean().round(1))
      }
    }
    with open('/home/nogaromo/Загрузки/ozon.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    file.close()
    

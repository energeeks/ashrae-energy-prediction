import requests
import pandas as pd
from flask import current_app


def get_forecast(lat, lon):
    call_root = "http://api.openweathermap.org/data/2.5/forecast"
    call_lat = "?lat="
    lat = str(lat)
    call_lon = "&lon="
    lon = str(lon)
    call_key = "&appid="
    api_key = current_app.config['API_KEY']

    call = [call_root, call_lat, lat, call_lon, lon, call_key, api_key]
    separator = ""
    call = separator.join(call)
    return requests.get(call)


def parse_request(request):
    request = request.json()
    request = request['list']

    main = []
    weather = []
    clouds = []
    wind = []
    date = []
    for r in request:
        main.append(r['main'])
        weather.append(r['weather'][0])
        clouds.append(r['clouds'])
        wind.append(r['wind'])
        date.append(r['dt_txt'])

    main = pd.DataFrame(main)
    weather = pd.DataFrame(weather)
    clouds = pd.DataFrame(clouds)
    clouds.columns = ["cloud_coverage"]
    wind = pd.DataFrame(wind)

    total = pd.concat([main, weather, clouds, wind], axis=1)
    total["date"] = pd.to_datetime(date)

    return total

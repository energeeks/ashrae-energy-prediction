import requests
import pandas as pd
from flask import current_app


def get_forecast(lat, lon):
    api_key = current_app.config['API_KEY']
    url = "http://api.openweathermap.org/data/2.5/forecast?&units=metric&lat={}&lon={}&appid={}".format(lat, lon, api_key)
    return requests.get(url)
    

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

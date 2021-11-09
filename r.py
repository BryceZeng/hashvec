import requests
import json
overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = "singapore 440055"
response = requests.get(overpass_url,
                        params={'data': overpass_query})
response

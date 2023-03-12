import requests
from requests.structures import CaseInsensitiveDict
from bs4 import BeautifulSoup
import json
import numpy as np

osm_groups = {
    'g_shop': """{
    "key": "group",
    "value": "g_shop"
    }""",
    'g_transport': """{
        "key": "group",
        "value": "g_transport"
    }""",
    'g_lodging': """{
        "key": "group",
        "value": "g_lodging"
    }""",
    'g_eat-out': """{
        "key": "group",
        "value": "g_eat-out"
    }""",
    'g_leisure': """{
        "key": "group",
        "value": "g_leisure"
    }""",
    'g_service': """{
        "key": "group",
        "value": "g_service"
    }""",
    'g_landmark': """{
        "key": "group",
        "value": "g_landmark"
    }""",
    'g_office': """{
        "key": "group",
        "value": "g_office"
    }""",
    'g_vehicle': """{
        "key": "group",
        "value": "g_vehicle"
    }""",
    'g_religion': """{
        "key": "group",
        "value": "g_religion"
    }"""
}

service_key = open('targomo_service_key.txt').read()


def get_group_node_count(group, lat, lng, walk_distance=1000):
  url = "https://api.targomo.com/pointofinterest/reachability"

  headers = CaseInsensitiveDict()
  headers["Content-Type"] = "application/json"

  data = """
  {
    "sources": [
      {
        "id": "1",
        "lat": """+str(lat)+""",
        "lng": """+str(lng)+""",
        "tm": {
          "walk": {}
        }
      }
    ],
    "elevation": true,
    "maxEdgeWeight":"""+str(walk_distance)+""",
    "edgeWeight": "distance",
    "osmTypes": ["""+group+"""
    ],
    "format": "geojson",
    "serviceUrl": "https://api.targomo.com/westcentraleurope/",
    "serviceKey": "BDVLQ87DL4MC63T3RANY"
  }
  """

  resp = requests.post(url, headers=headers, data=data)
  parsed = json.loads(resp.text)

  # print json response
  # response_json = json.dumps(parsed, indent=4)
  # print(response_json)

  return len(parsed['features'])

def get_geographical_context_vector(lat, lon):
  vector = []
  for group in osm_groups:
    vector.append(get_group_node_count(osm_groups[group], lat, lon))

  normalized_vector = [round(x / max(vector), 2) for x in vector]
  return np.asarray(normalized_vector)

# get_geographical_context_vector(52.511720, 13.322269)
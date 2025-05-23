import os
import requests
import json

credentials = json.loads('creds.jon')

print(credentials)

#api_key = os.environ.get("API_KEY")
#api_url = "https://api.example.com/data"


#headers = {"Authorization": f"Bearer {api_key}"}

#response = requests.get(api_url, headers=headers)
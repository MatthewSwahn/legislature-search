import requests
import os
from dotenv import load_dotenv

load_dotenv()

def download_bill_xml():
    '''downloads the XML used for this repo'''
    url = "https://www.congress.gov/119/bills/hr1/BILLS-119hr1enr.xml"
    filename = os.getenv('bills-119hr1enr')
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully downloaded {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    download_bill_xml()
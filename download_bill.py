import requests
import os

def download_bill_xml():
    url = "https://www.congress.gov/119/bills/hr1/BILLS-119hr1enr.xml"
    filename = os.path.join("data", "BILLS-119hr1enr.xml")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
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
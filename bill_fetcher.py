import requests
import json
from typing import List
import time

def get_bill_info(congress_num:int, bill_type:str, bill_number:int, api_key:str)-> dict:
    '''
    if given information about a bill, will return a flat dictionary about the bill
    
    Args:
            congress_num (int): what session of congress
            bill_type (str): Bill/Amendmentent and from the house or senate
            bill_number (int): bill identification number
            api_key (str): data.gov api key
    
    Returns:
        dict: containing the text_url
    '''
    bill_url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}/{bill_number}"
    headers = {"X-API-Key": api_key}
    bill_response = requests.get(bill_url, headers=headers).json()['bill']
    
    title = bill_response['title']
    
    # create dictionary response and return
    response = dict()
    response['congress_num'] = congress_num
    response['bill_type'] = bill_type
    response['bill_number'] = bill_number
    response['title'] = title
    
    # get text url
    bill_text_url = bill_response['textVersions']['url']
    response['bill_text_url'] = bill_text_url
    
    response['bill_text'] = _get_bill_text(bill_text_url=bill_text_url, api_key=api_key)
    
    # get amendment urls
    amendments_details_url = bill_response['amendments']['url']
    response['amendments_details_url'] = amendments_details_url
    
    return response

def _get_bill_text(bill_text_url:str, api_key:str) -> dict:
    '''
    once given the bill info, will return a similar dictionary but with the text attached
    
    Args:
            congress_num (int): what session of congress
            bill_type (str): Bill/Amendmentent and from the house or senate
            bill_number (int): bill identification number
            api_key (str): data.gov api key
    
    Returns:
        dict: containing the text_url
    '''
    headers = {"X-API-Key": api_key}
    bill_text_source_response = requests.get(bill_text_url, headers=headers).json()
    
    most_recent_response_formats= bill_text_source_response['textVersions'][0]['formats']
    for format in most_recent_response_formats:
        if format['type'] == 'Formatted Text':
            bill_text_url = format['url']
    
    bill_text_response = requests.get(bill_text_url)
    
    return bill_text_response.text


def get_amendment_info(congress_num:int, bill_type:str, bill_number:int, api_key:str) -> List[dict]:
    '''
    once given the bill info, will return list containing dictionaries of all related amendments
    Args:
            congress_num (int): what session of congress
            bill_type (str): Bill/Amendmentent and from the house or senate
            bill_number (int): bill identification number
            api_key (str): data.gov api key
    
    Returns:
        List[dict]: a list containing each amendment payload
    '''
    amendment_detail_url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}/{bill_number}/amendments"
    headers = {"X-API-Key": api_key}
    
    all_amendments = []
    url = amendment_detail_url

    while url:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        amendments = data.get("amendments", [])
        
        if len(amendments) > 0:
            amendments_with_text = list()
            for amendment in amendments:
                congress_num = amendment['congress']
                amend_type = amendment['type'].lower()
                amend_number = amendment['number']
                amendment_text = _get_amendment_text(congress_num=congress_num, amend_type=amend_type, amendment_number=amend_number, api_key=api_key)
                updated_amendment = amendment.copy()
                updated_amendment['amend_text'] = amendment_text
                amendments_with_text.append(updated_amendment)
            
            amendments = amendments_with_text
        
        all_amendments += amendments
        # Get the next page URL or set to None if not present
        url = data.get("pagination", {}).get("next")


    return all_amendments

def _get_amendment_text(congress_num:int, amend_type:str, amendment_number:int, api_key:str) -> str:
    '''
    if given a list containing dictionaries of all related amendments, will return that list and add a new key "amend_text", that contains the 
    
    Args:
            congress_num (int): what session of congress
            amend_type (str): amendment category
            amendment_number (int): amendment identification number
            api_key (str): data.gov api key
    
    Returns:
        amendment text
    '''
    results = ''
    headers = {"X-API-Key": api_key}
    
    amend_text_url = f"https://api.congress.gov/v3/amendment/{congress_num}/{amend_type}/{amendment_number}/text"
    amendment_overview = requests.get(amend_text_url, headers=headers).json()
    
    if amendment_overview.get('textVersions'):
        amendment_formats = amendment_overview['textVersions'][0]['formats']
        for format in amendment_formats:
            if format['type'] == 'HTML':
                amendment_text_url = format['url']
        amendment_text_response = requests.get(amendment_text_url)
        
        results = amendment_text_response.text
        
        # if timed out, wait 3 minutes and try again
        while results[:73] == '<!DOCTYPE html><html lang=\"en-US\"><head><title>Just a moment...</title>':
            time.sleep(360)
            results = _get_amendment_text(congress_num=congress_num, amend_type=amend_type, amendment_number=amendment_number, api_key=api_key)
            
        
    return results
import requests
import json


def get_bill_info(congress_num:int, bill_type:str, bill_number:int, api_key:str)-> dict:
    '''
    if given information about the bill, will return a flat dictionary about the bill
    
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
    text_url = bill_response['textVersions']['url']
    amendments_details_url = bill_response['amendments']['url']
    
    # create dictionary response and return
    response = dict()
    response['congress_num'] = congress_num
    response['bill_type'] = bill_type
    response['bill_number'] = bill_number
    response['title'] = title
    response['text_url'] = text_url
    response['amendments_details_url'] = amendments_details_url
    
    return response

def get_bill_text(bill_info:dict, api_key:str) -> dict:
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
    bill_text_url = bill_info['text_url']
    headers = {"X-API-Key": api_key}
    bill_text = requests.get(bill_text_url, headers=headers).text
    
    response = bill_info.copy()
    response['bill_text'] = bill_text
    
    return response
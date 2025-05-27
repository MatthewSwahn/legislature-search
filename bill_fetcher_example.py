from bill_fetcher import get_bill_info, get_bill_text, get_amendment_info, get_amendment_text
import json
import requests
import os

def main():
    with open('creds.json', 'r') as f:
        credentials = json.load(f)

    api_key = credentials.get('api_gov_key')
    congress_num = 119
    # budget is hconres 14, one big beautiful bill is hr 1
    bill_type = "hconres"
    bill_number = 14
    
    # save all data to directory
    dir_path = f'data/congress_{congress_num}_{bill_type}{bill_number}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    bill_info = get_bill_info(congress_num=congress_num,
                              bill_type=bill_type,
                              bill_number=bill_number,
                              api_key=api_key)
    
    bill_info = get_bill_text(bill_info=bill_info,
                              api_key=api_key)
    
    # amendment info
    amendment_info = get_amendment_info(congress_num=congress_num,
                              bill_type=bill_type,
                              bill_number=bill_number,
                              api_key=api_key)
    
    amendment_info = get_amendment_text(amend_info_list=amendment_info,
                                        api_key=api_key)
    
    
    # write bills and amendment data to directory
    bill_json_path = os.path.join(dir_path, f'bill_data_{congress_num}_{bill_type}{bill_number}')
    if not os.path.exists(bill_json_path):
        with open(bill_json_path, 'w') as f:
            json.dump(bill_info, f, indent=4)
            
    amendment_json_path = os.path.join(dir_path, f'amendments_data_for_bill_{congress_num}_{bill_type}{bill_number}')
    if not os.path.exists(amendment_json_path):
        with open(amendment_json_path, 'w') as f:
            json.dump(amendment_info, f, indent=4)
            
    
if __name__=='__main__':
    main()
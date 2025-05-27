from bill_fetcher import get_bill_info, get_bill_text, get_amendment_info, get_amendment_text
import json
import requests

def main():
    with open('creds.json', 'r') as f:
        credentials = json.load(f)

    api_key = credentials.get('api_gov_key')
    congress_num = 119
    # budget is hconres 14, one big beautiful bill is hr 1
    bill_type = "hconres"
    bill_number = 14

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
    
if __name__=='__main__':
    main()
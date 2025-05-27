from bill_fetcher import get_bill_info, get_bill_text
import json


def main():
    with open('creds.json', 'r') as f:
        credentials = json.load(f)

    api_key = credentials.get('api_gov_key')
    congress_num = 119
    bill_type = "hconres"
    bill_number = 14

    bill_info = get_bill_info(congress_num=congress_num,
                              bill_type=bill_type,
                              bill_number=bill_number,
                              api_key=api_key)
    
    print('bill info', bill_info)
    
    bill_text = get_bill_text(bill_info=bill_info,
                              api_key=api_key)
    
    print('bill text', bill_text.get('bill_text'))

if __name__=='__main__':
    main()
# Import Libraries
import requests
from bs4 import BeautifulSoup
import os 

def get_captcha(index):
    try:
        # Step 1: Request the main URL
        base_url = 'https://www.callofduty.com/redemption'
        response = requests.get(base_url)
    
        # check if the request has succeeded.
        if response.status_code==200:
            # Step 2: Parse the source code
            html_doc = response.text
            soup = BeautifulSoup(html_doc, 'html.parser')       
        
            # Step 3: Extract the captcha URL
            captcha_url = soup.find('img', id='val_code').get('src')
        
            # Step 4: Download the captcha Image
            captcha_image = requests.get(captcha_url)
            items = os.listdir(f'{os.getcwd()}/downloaded_images')

            if index < len(items):
                file_name = f'captcha_{len(items)}.png'
                file_path = os.path.join('./downloaded_images', file_name)
            else:
                file_name = f'captcha_{index}.png'
                file_path = os.path.join('./downloaded_images', file_name)

            with open(file_path , 'wb') as file:
                file.write(captcha_image.content)
                print(f'Captcha image saved as {file_name}')
            
    except Exception as e:
        print(f"error raised! the status code is {response.status_code}.")
        print(f'eror: {e}')






# Import Libraries
import os 
import requests
import concurrent.futures
from bs4 import BeautifulSoup

def get_captcha(index):
    #try:
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

            captcha_url = 'https://west-mrms.codm.activision.com/commonAct/codmwest_cdk_center/valCode.php?sServiceType=codmatvi&codeKey=ca77d4727176664cf187f3d03e67f45f&_rand=0.34989549791903407'
            
            # Step 4: Download the captcha Image
            captcha_image = requests.get(captcha_url)
            items = os.listdir(f'{os.getcwd()}/test_images')

            if f'captcha_{index}.png' in os.listdir('./test_images'):
                pass
            else:
                file_name = f'captcha_{index}.png'
                file_path = os.path.join('./test_images', file_name)

            with open(file_path , 'wb') as file:
                file.write(captcha_image.content)
                print(f'Captcha image saved as {file_name}')
            
    #except Exception as e:
     #   print(f'eror: {e}')
      #  print(f"error raised! the status code is {response.status_code}.")


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
     executor.map(get_captcha, range(100))
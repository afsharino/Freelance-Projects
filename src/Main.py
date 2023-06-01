from Scraper import get_captcha
from tqdm.notebook import tqdm 

if __name__ == '__main__':
    for i in tqdm(range(3000)):
        get_captcha(i)
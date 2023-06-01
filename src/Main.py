from Scraper import get_captcha
import concurrent.futures

if __name__ == '__main__':
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
     executor.map(get_captcha, range(2))
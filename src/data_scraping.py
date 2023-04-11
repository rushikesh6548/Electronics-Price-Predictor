import time
import bs4
import requests
from bs4 import BeautifulSoup as bs
from src.exception import CustomException
from src.logger import logging
import html_extractor  # Class where mobile_html_extractor is written
import os


class DataScraping:
    def __init__(self):
        self.mobile_url = 'https://www.smartprix.com/mobiles'
        self.tablets_url = 'https://www.smartprix.com/tablets'
        self.laptops_url = 'https://www.flipkart.com/search?q=laptop&otracker=search&otracker1=search&marketplace=FLIPKART&as-show'

    def extract_mobile_html(self):
        logging.info("Entered extract_mobile_html method or component")

        mobile_html_ext_obj = html_extractor.HtmlExtractor(main_url=self.mobile_url,
                                                           button_path='//*[@id="app"]/main/div[1]/div[2]/div[3]')
        mobile_html_ext_obj.do_clicks(click_path='//*[@id="app"]/main/aside/div/div[5]/div[2]/label[1]/input')
        html_filename = 'available_'
        path = os.path.join('src','data','html_files', 'mobile_html')  #src/data/html_files
        if not os.path.exists(path):
            os.makedirs(path)
        mobile_html_ext_obj.extract_entire_html(html_filename=os.path.join(path, html_filename))
        logging.info("Mobile html extraction completed")



    def extract_tablets_html(self):
        logging.info("Entered extract_tablets_html method or component")
        try:
            tab_extr_obj = html_extractor.Smartprix_HtmlExtractor(self.tablets_url,
                                                                  button_path='//*[@id="app"]/main/div[1]/div[2]/div[3]')
            tab_extr_obj.do_clicks(click_path='//*[@id="app"]/main/aside/div/div[5]/div[2]/label[2]')
            html_filename = 'all_tablets.html'
            path = os.path.join('html_files', 'tablets_html')
            if not os.path.exists(path):
                os.makedirs(path)
            tab_extr_obj.extract_entire_html(html_filename=os.path.join(path, html_filename))
            logging.info("Tablets html extraction completed")

        except Exception as e:
            raise CustomException(e)

    def extract_laptop_html(self,html_filename):
        logging.info("Entered extract_laptop_html method or component")
        try:
            path = os.path.join('data','html_files', 'laptop_html')
            if not os.path.exists(path):
                os.makedirs(path)
            for i in range(1, 41):
                link = f"{self.laptops_url}&page={i}"
                res = requests.get(link)
                soup = bs(res.content, 'html.parser')
                html_filename = f"laptops_flipkart_{i}.html"
                with open(os.path.join(path, html_filename), "w", encoding="utf-8") as f:
                    f.write(soup.prettify())
            logging.info("Laptop html extraction completed")

        except Exception as e:
            raise CustomException(e)

        for i in range(1, 26):
            link = f'https://www.flipkart.com/search?q=laptop&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&p%5B%5D=facets.price_range.from%3D20000&p%5B%5D=facets.price_range.to%3D60000&sort=price_asc&page={i}'
            page = requests.get(link)
            html_content = bs(page.content, 'html.parser')
            html_content_str = str(html_content)
            with open(f'{i}_flipkart_laptops_set2.html', 'w', encoding='utf-8') as f:
                f.write(html_content_str)
            logging.info('Laptop html set 2 extraction completed')


if __name__ == "__main__":
    obj = DataScraping()
    print("d12")
    obj.extract_mobile_html()
    obj.extract_tablets_html()
    obj.extract_laptop_html() # to extract all laptop !

    print("d1")

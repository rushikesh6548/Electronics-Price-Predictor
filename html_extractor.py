import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By



class HtmlExtractor:
    def __init__(self,main_url,button_path):
        self.main_url = main_url
        self.button_path = button_path
        s = Service('D:/chromedriver.exe')
        self.driver = driver = webdriver.Chrome(service=s)
        driver.get(main_url)


    def do_clicks(self,click_path):
        self.driver.find_element(by=By.XPATH,value=click_path).click()


    def extract_entire_html(self,html_filename):
        old_height = self.driver.execute_script('return document.body.scrollHeight')


        # Logic behind this : As our page needs to be clicked on Load More button to load the next mobile phones, we click the button until the point that
        # the old height[orignal start page height] becomes same as the new height[when we reach at the bottom] as then there will be no further items !
        while True:
            self.driver.find_element(by = By.XPATH, value=self.button_path).click()
            time.sleep(2)

            new_height = self.driver.execute_script('return document.body.scrollHeight')

            if new_height == old_height:
                break

            old_height = new_height

        html = self.driver.page_source

        with open(html_filename,'w',encoding='utf-8') as f:
            f.write(html)



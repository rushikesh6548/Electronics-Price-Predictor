import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipelinelaptop


class PricePredictApp:
    def __init__(self):
        self.current_page = 'home'
        self.home()


    def home(self):
        st.markdown("""<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>""",
                    unsafe_allow_html=True)

        st.title("Electronics Price Prediction App")
        st.write("This app predicts electronics item prices using Machine learning ! ")
        st.write("The complete code for this can be found here : ")
        menu_selection = st.sidebar.selectbox(
            "Select an Electronics item  to predict price of :",
            ("Laptop", "Smartphone")
        )

        if menu_selection == "Laptop":
            self.laptop_page()
        elif menu_selection == "Smartphone":
            self.smartphone_page()

    def laptop_page(self):
        st.empty()
        st.header("Laptop Price Prediction")
        # Add your laptop-related UI elements here
        brand_categories = ['ASUS', 'MSI', 'APPLE', 'DELL', 'HP', 'LENOVO', 'ACER', 'GIGABYTE',
       'MICROSOFT', 'INFINIX', 'LG', 'SAMSUNG', 'AVITA', 'REALME', 'VAIO',
       'MI', 'SMARTRON', 'REDMIBOOK', 'NOKIA']
        brand_name = st.selectbox('Brand:',brand_categories)

        model_categories = ['ROG', 'X', 'MACBOOK', 'ALIENWARE', 'ZENBOOK', 'ALLIENWARE', 'OMEN', 'STEALTH', 'PROART',
                            'THINKPAD', 'RAIDER', 'PREDATOR', 'CREATOR', '15S', 'GS66', 'AERO', 'LEGION', 'RYZEN',
                            'GP76', 'YOGA', 'EXPERTBOOK', 'SPECTRE', 'T14', 'WF65', 'TUF', 'THINPAD', 'GP65', 'G15',
                            'THINKBOOK', 'INTEL', 'SWIFT', 'PULSE', 'SUMMIT', 'SPIN', 'GP66', 'IDEAPAD', 'VICTUS',
                            'ZEPHYRUS', 'XPS', 'ENVY', 'INSPIRON', 'CROSSHAIR', 'NITRO', 'VIVOBOOK', '7000', 'HP',
                            'SURFACE', 'ALPHA', 'G5', 'ZERO', 'PAVILION', 'PRO', 'GF65', 'CONCEPTD', 'GRAM', 'GALAXY',
                            'SWORD', 'ASUS', 'KATANA', 'ASPIRE', 'MODERN', 'PRESTIGE', 'LATITUDE', 'VOSTRO', 'DELL',
                            '5000', '14S', 'CHROMEBOOK', '3511', 'LIBER', 'GL', 'BOOK(SLIM)', 'SE', '15-EC1105AX',
                            'INBOOK', 'GF63', 'PROBOOK', 'NOTEBOOK', 'BRAVO', 'E', 'T.BOOK', 'G8', 'X1', 'B50-70',
                            'BOOK', 'TRAVELMATE', '240', '15', 'EXTENSA', 'PUREBOOK', 'INSPRION', 'V15', '250', 'G',
                            'V14', 'ATHLON', '255', 'EEEBOOK', 'PENTIUM', '247', 'PURA', 'LENOVO', 'MEDIATEK',
                            'CELERON', '(2022)', 'APU', 'ONE']

        model_name = st.selectbox('Model[If no particular Model Name , Select X ]', model_categories)

        processor_categories = ['AMD', 'INTEL', 'APPLE', 'QUALCOMM', 'MEDIATEK']
        processor_brand = st.selectbox('Processor Brand:', processor_categories)

        processor_model_categories = ['RYZEN9', 'COREI9', 'M2MAX', 'COREI7', 'M1MAX', 'RYZEN7', 'M2PRO', 'M1PRO',
                                      'COREI5', 'M2PROCESSOR', 'M1PROCESSOR', 'RYZEN5', 'COREI3', 'RYZENR5', 'RYZEN3',
                                      'SNAPDRAGON7C', 'CELERONDUAL', 'ATHALON', 'PENTIUMQUAD', 'ATHLONDUAL', 'PQC',
                                      'PENTIUMSILVER', 'MEDIATEKKOMPANIO', 'CELERONQUAD', 'DUALCORE', 'APUDUAL',
                                      'PENTIUMGOLD']
        processor_model = st.selectbox('Processor Model:', processor_model_categories)

        ram_size_categories = [32, 16, 8, 4, 64]
        ram_size = st.selectbox('Ram Size [In GB]',ram_size_categories)

        ram_type_categories  = ['DDR5', 'Unified', 'LPDDR5', 'DDR4', 'DDR3', 'LPDDR4X', 'LPDDR3',
       'LPDDR4']
        ram_type = st.selectbox('Ram Type:', ram_type_categories)

        ssd_size_gb_categories = [2048, 4096, 1024, 512, 256, 128, 0, 8]
        ssd_size = st.selectbox('SSD Size [IN GB]:',ssd_size_gb_categories)

        hdd_size_gb_categories = [0, 1024, 2048, 512, 256, 500]
        hdd_size = st.selectbox('HDD Size [IN GB]:',hdd_size_gb_categories)

        display_size_categories = [17., 14., 15.6, 13.3, 12., 10., 11.6]
        display_size = st.selectbox('Screen Size [IN Inches]:',display_size_categories)

        operating_sys_categories = ['Windows 11', 'Mac', 'Windows 10', 'Linux', 'Chrome']
        operating_sys = st.selectbox('Operating System:',operating_sys_categories)

        warranty_categories = [1., 2., 3., 0., 1.5]
        warranty = st.selectbox('Warranty Period [In Years]:',warranty_categories)

        touchscreen = st.selectbox('Touchscreen?[Yes or NO]',['Yes','No'])

        graphics_size_categories = [16, 12, 8, 6, 10, 2, 4, 0]
        graphics_size = st.selectbox('Graphics Size[In GB]', graphics_size_categories)

        if st.button('Predict Price'):
            if touchscreen == "Yes":
                touchscreen = 1
            else:
                touchscreen = 0

            # brand_name,model_name,processor_brand,processor_model,price,ram_size,ram_type,ssd_size_gb,hdd_size_gb,display_size,operating_sys,warranty,touchscreen,graphics_size

            input_feats_as_df = pd.DataFrame([{
                'brand_name': brand_name,
                'model_name': model_name,
                'processor_brand': processor_brand,
                'processor_model': processor_model,
                'ram_type': ram_type,
                'operating_sys': operating_sys,
                'ram_size': ram_size,
                'ssd_size_gb': ssd_size,
                'hdd_size_gb': hdd_size,
                'display_size': display_size,
                'warranty': warranty,
                'touchscreen': touchscreen,
                'graphics_size': graphics_size
            }])



            predict_obj = PredictPipelinelaptop(features=input_feats_as_df)

            prediction= predict_obj.predict()

            st.title(f"The estimated price of this Laptop configuration will be Rs: {round(prediction[0])}")





    def smartphone_page(self):
        st.empty()
        st.header("Smartphone Price Prediction")
        # Add your smartphone-related UI elements here





if __name__ == '__main__':
    app = PricePredictApp()

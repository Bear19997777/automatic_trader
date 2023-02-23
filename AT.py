import pandas as pd
import shioaji as sj 
from datetime import datetime as dt
import requests
from datetime import timezone
import numpy as np
from dotenv import load_dotenv
import os 
class at:
    def __init__   (self):
        load_dotenv()
        api = sj.Shioaji(simulation=True)
        self.__api = api
        
    def login(self):
        self.__api.login(
            api_key=f'{os.getenv("api_key")}',
            secret_key=f'{os.getenv("secrect_key")}'
            )
          
    def main(self):
        self.login()
if __name__ == "__main__":
    atobj = at()
    atobj.main()



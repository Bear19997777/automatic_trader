import pandas as pd
import shioaji as sj 
from datetime import datetime as dt
import requests
from datetime import timezone
import numpy as np
from dotenv import load_dotenv
import os 
load_dotenv()
api = sj.Shioaji(simulation=True)
print(os.getenv("api_key"))
print(os.getenv("secrect_key"))
api.login(
    api_key=f'{os.getenv("api_key")}',
    secret_key=f'{os.getenv("secrect_key")}'
          
)


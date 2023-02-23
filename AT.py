import pandas as pd
import shioaji as sj 
from datetime import datetime as dt
import requests
from datetime import timezone
import numpy as np
import dotenv as dt 

api = sj.Shioaji(simulation=True)
api.login(
    api_key='',
    secret_key=""
          
)

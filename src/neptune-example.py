import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('NEPTUNE_API_TOKEN')

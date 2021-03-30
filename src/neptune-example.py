import os
from dotenv import load_dotenv
import neptune

load_dotenv()

TOKEN = os.getenv('NEPTUNE_API_TOKEN')

neptune.init('marcinknara/Neptune-Example', api_token=TOKEN)
neptune.create_experiment('first-experiment')

neptune.append__tag('great-idea-1')

#code runs here (usually training and validation)

# log some metric values
neptune.log_metric('roc_auc', 0.93)

#more code


neptune.set_property('status', 'completed with no issues')




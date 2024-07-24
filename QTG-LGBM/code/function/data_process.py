import re
import os
def clean_feature_name(name):
    name = re.sub(r'[^\w]', '_', name)
    name = name.strip('_')
    return name


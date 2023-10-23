import re


def reg_text(text):
    text = re.sub('Ã‚â‚¬Ã‚Â', '', text)
    text = re.sub('Ã‚â‚¬Ã‚â', '', text)
    text = re.sub('Ã‚â‚¬Ã‚Â¦', '', text)
    text = re.sub('Ã‚â‚¬Ã‚Âº', '', text)
    text = re.sub('Ã‚â‚¬Ã‚â', '', text)
    text = re.sub('ÃƒÂ¯Ã‚â', '', text)
    text = re.sub('Ã‚â‚¬Ã', '', text)
    text = re.sub('Ã‚â', '', text)
    text = re.sub('šÃ‚Â', '', text)
    text = re.sub('ÃƒÅ½Ã', '', text)
    text = re.sub('šÃ‚Â§', '', text)
    text = re.sub('ÃƒÂ', '', text)
    text = re.sub('œÃƒÂ', '', text)
    text = re.sub('šÃ‚Â', '', text)
    text = re.sub('„¢', '', text)
    text = re.sub('¢', '', text)
    text = re.sub('œ', '', text)
    text = re.sub('€', '', text)
    text = re.sub(' +', ' ', text)
    return text
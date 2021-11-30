from deep_translator import GoogleTranslator
from deep_translator import single_detection , batch_detection

lang = batch_detection(['bonjour la vie', 'hello world'], api_key='your_api_key')
print(lang) # output: [fr, en]

translated = GoogleTranslator(source='auto', target=lang).translate(text=text)

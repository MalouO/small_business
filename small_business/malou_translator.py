from google_trans_new import google_translator
import pandas as pd
#note source code in google_trans_new needs to be updated. Ask me how!

def translate(df):
    for i in df.Review:
        translator = google_translator()
        i = translator.translate(i, lang_src = "auto", lang_tgt = "en")
        df.Review[i] = i

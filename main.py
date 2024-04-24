
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
from gpt import GPT, FirstNotificationOfLoss
import pandas as pd
import time
from loguru import logger


context = """

                    Sehr geehrte Damen und Herren\nnachfolgend erhalten \

                    Sie eine Schadenanzeige unseres Kunden. Im Anhang Bilder vom\nSchaden\nBitte teilen Sie uns die \

                    Schadennummer mit und leiten sämtlichen Schriftwechsel in\nKopie an uns weiter.\nKunden- \

                    Vertragsdaten\nVS-Nr. 2-22.955.000-2 Produkt Gebäude\nVN: Sarah, \

                    Straße: Sternstraße\Mayer 99\nPLZ: 90053 Ort: Nürnberg\nSchadendaten\nSchadentag: 25.01.2024 \

                    Schaden-Ort: Sternstraße 99, 90053\nNürnberg -unbekannt\nSchadenhergang: Sehr geehrte Damen und \

                    Herren,\nhiermit melde ich einen Schaden.\nVorgestern am 25.01.2024 gab es ein Sturm und heute \

                    habe ich gemerkt, dass\neine Scheibe im Glaszaun gesprungen ist. Es sieht so aus, als ob es von \

                    einem\nStein getroffen wurde.

                    Der Boden im Garten ist mit Kieselsteinen belegt.\nBilder anbei\nSchadenumfang:\nSchuldfrage:\

                    unbekannt Bei Polizei gern.:\nTagebuch Nr.:\nGeschätzte Gutachter notw.:\nHöhe:\nSelbstbeteiligung \

                    0 ? Abrechnung: 1Brutto\nBemerkung:\nAnspruchsteller-Daten\nSarah, Mayer Straße: Sternstraße 99 \

                    Telefon: 7896/8910373E-Mail: ml@heilpraktikerinnuernberg.d\nTelefon\nMobil: 38271/378127PLZ: 90053 \

                    Ort: Nürnberg\nIBAN DE55 9821 7839 3719 5262 12 Kontoinhaber Sarah, Mayer\n2\nMit freundlichen Grüßen\

                    Best regards\nRenna Wahlt
                    """
                  
# claims_type = gpt.get_claims_type(context)
# print(claims_type)
# cause_type = gpt.get_cause(context, claims_type=claims_type)
# print(cause_type)
# notifier = gpt.get_notifier(context)
# print(notifier)
# date = gpt.get_claims_date(context)
# print(date)
# claims_object = gpt.get_claims_objekt(context)
# print(claims_object)

# print(gpt.fde_first_notification_of_loss(context))

df = pd.read_parquet("df_dirk_date_anon.parquet")

start_time = time.time()

logger.info(f"Operation Started at {start_time}")

logger.info(f"connect to OpenAI")
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
gpt = GPT(client=client)
logger.success("Connected to OpenAI with Key")

doc = df.iloc[3]['text'] 

logger.info("Starting predictions")
claim = gpt.fde_first_notification_of_loss(doc)
logger.success(f"Predictions done:{claim}")

end_time = time.time()
total_time = end_time - start_time
logger.success(f"Operation done at {end_time}")

logger.info(f"Processing Time: {total_time}")

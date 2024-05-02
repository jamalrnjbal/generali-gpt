
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
from gpt import GPT, FirstNotificationOfLoss, group_sd_urs_art, DirksClaims
import pandas as pd
import time
from loguru import logger
from collections import Counter



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
     
     
# logger.info(f"Operation Started at {start_time}")

logger.info(f"connect to OpenAI")
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
gpt = GPT(client=client)
logger.success("Connected to OpenAI with Key")             
# claims_type = gpt.get_claims_type(context)
# print(claims_type)
# cause_type = gpt.get_cause(context, claims_type=claims_type)
# print(cause_type)
# notifier = gpt.get_notifier(context)
# print(notifier)
date = gpt.get_claims_date(context)
print(date)
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

logger.info("Preparing SD-URS-ART")

df = group_sd_urs_art(df.copy())
logger.info("Preparing Schaden-Datum")
df["schadentag"] = df["schadentag"].apply(lambda x: x.replace("-", "."))

hive_data = DirksClaims(
        doc_id_list=df["doc_id"].to_list(),
        schaden_objekt_list=df["schaden_objekt"].to_list(),
        schaden_typ_list=df["sd_typ_kennung"].to_list(),
        sd_urs_art_list=df["sd_urs_art"].to_list(),
        schaden_datum_list=df["schadentag"].to_list(),
    ).info

logger.success("Hive Data Prepared")

logger.info("Big Evaluation")
start_time = time.time()

gpt_predictions = [
        GPT(client).fde_first_notification_of_loss(
            document=text
        )
        for text in df["text"]
    ]

end_time = time.time()
total_time = end_time - start_time
logger.info(f"Mass Processing Time: {total_time}")

logger.info("Start Preparing Predictions for Comparison")

doc_ids = df["doc_id"].to_list()
gpt_predictions = [
    {**cl, "doc_id": doc_id}
    for doc_id, cl in zip(doc_ids, gpt_predictions)
]

preds = [
        FirstNotificationOfLoss(
            doc_id=cl["doc_id"],
            schaden_objekt=cl["objekt"],
            schaden_typ=cl["type"],
            sd_urs_art=cl["cause"],
            schaden_datum=cl["date"],
        )
        for cl in gpt_predictions
    ]

logger.info("Preparation for Comparison completed")

match = [i == j for i, j in zip(preds, hive_data)]

logger.info(f"Results for all fields: {Counter(match)}")

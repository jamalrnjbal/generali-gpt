import json
from loguru import logger
from dataclasses import dataclass
import pandas as pd
from typing import List
from collections import Counter

class GPT:
    def __init__(self, client):
        """Initialize a GPT instance with a given version."""
        self.client = client
        self.cause_mapping = {
            "LW": {
                "Sonstiges": 0,
                "Rohrbruch": 2,
                "Armaturen": 4,
                "Fehlverhalten": 9,
            },
            "ST": {
                "Sturm sonstiges": 10,
                "Sturm Grundstücksbestandteile": 19,
                "Hagel": 4,
                "Gartenmöbel": 7,
            },
            "EL": {
                "Überflutung durch Starkregen": 2,
                "Sonstiges": 0,
            },
            "ED": {
                "Sonstiges": 0,
                "einfacher Diebstahl": 70,
                "Fahrraddiebstahl": 79,
                "Vandalismus": 9,
            },
            "FE": {
                "Sonstiges": 0,
                "Überspannung": 6,
                "Fehlverhalten": 9,
            },
        }
        self.special_cases = {("GL", "GL"): 4711}
        self.prompt = """
        Du bist ein intelligenter Assistent in der Schadenbearbeitung einer \
        Versicherung. Bitte lies das nachfolgende Dokument aufmerksam und führe \
        anschließend die Anweisungen exakt durch.\
        Bitte geben Sie die Antwort in einem json Format zurück.
        """
    
    def get_claims_type(self, document:str) -> str:
        instructions = """
        - Beantworte, unter Verwendung der entsprechenden Kategorie, in welche \
        Schaden-Kategorie das obige Dokument am besten klassifierziert werden kann.
        Kategorien: [Leitungswasser, Sturm, Feuer, Elementar, Diebstahl, Glass,
        Sonstige]
        - Beantworte die Frage in dem Du ausschließlich das Kürzel der jeweiligen \
        Kategorie ohne weitere Leerzeichen oder Sonderzeichen zurückgibst:  \
        Kürzel: Leitungswasser = 'LW', Sturm = 'ST', Feuer = 'FE', \
        Elementar = 'EL', Diebstahl = 'ED', Glass = 'GL', Sonstige = 'Other'
        """
        
        messages=[
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": document},
            {"role": "user", "content": instructions}
        ]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=messages
        )
        
        claims_type_dict = json.loads(response.choices[0].message.content)
        claims_type = list(claims_type_dict.values())[0]
        
        return claims_type
    
    def get_cause(self, document: str, claims_type: str) -> str:
        """
        Classifies the type of an insurance claim based on the content of the
            document.

            This method generates a prompt for the language model using a specific
            template, sends the prompt to the model, and returns the model's
            response. The response classifies the claim into predefined categories
            based on the document content.
        
        """
        
        if claims_type == "LW":
            cause_types = [
                "Rohrbruch",
                "Armaturen",
                "Fehlverhalten",
                "Sonstiges",
            ]
        elif claims_type == "ST":
            cause_types = [
                "Sturm Grundstücksbestandteile",
                "Sturm sonstiges",
            "Hagel",
                    "Gartenmöbel",
                ]
        elif claims_type == "EL":
            cause_types = [
                "Überflutung durch Starkregen",
                "Sonstiges",
            ]
        elif claims_type == "ED":
            cause_types = [
                "einfacher Diebstahl",
                "Fahrraddiebstahl",
                "Vandalismus",
                "Sonstiges",
            ]
        elif claims_type == "GL":
            cause_types = [
                "Display Schaden",
                "Einfachverglasung",
                "Sonderverglasung",
                "Sonstiges",
            ]
        elif claims_type == "FE":
            cause_types = [
                "Überspannung",
                "Fehlverhalten",
                "Sonstiges",
            ]
        else:
            logger.info(
                f"Schaden-Typ: {claims_type}, "
                f"daher wird keine SD-URS-ART ermittelt"
            )
            return None
        
        instructions = f"""
            Dieses Dokument wurde in die Schaden-Kategorie '{claims_type}' 
            klassifiziert.
            
            <instructions>
                - Spezifiziere die zugehörige Schaden-Ursache aus den folgenden
                    Kategorien: {cause_types}
                - Beantworte die Frage in dem Du ausschließlich den Begriff aus 
                    der Liste ohne weitere Leerzeichen zurückgibst
            </instructions>
            """
            
        messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": document},
                {"role": "user", "content": instructions}
            ]
        
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                response_format={ "type": "json_object" },
                messages=messages
            )
            
        cause_type_dict = json.loads(response.choices[0].message.content)
        cause_type = list(cause_type_dict.values())[0]
        
        return cause_type
    
    def get_notifier(self, document: str) -> str:
        
        """
        Determines who reported the insurance claim based on the document
        content.

        This method uses a specific template to generate a prompt for the
        language model.The prompt asks the model to identify who reported
        the damage from the content of the document. The response
        categorizes the notifier into predefined groups: policyholder,
        field agent, or others.
        """
        
        instructions = """
                <instructions>
                - Beantworte, unter Verwendung der entsprechenden Kategorie, wer
                den Schaden gemeldet hat Kategorien: 
                [Versicherungsnehmer, Außendienstler, Sonstige]
                - Beantworte die Frage in dem Du ausschließlich das Kürzel der 
                jeweiligen Kategorie ohne weitere Leerzeichen oder Sonderzeichen 
                zurückgibst:  Kürzel: Versicherungsnehmer = 'VN', 
                Außendienstler = 'AD', Sonstige = 'Other' 
                </instructions>
                """
                            
        messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": document},
                {"role": "user", "content": instructions}
            ]
        
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                response_format={ "type": "json_object" },
                messages=messages
            )
            
        notifier_dict = json.loads(response.choices[0].message.content)
        notifier = list(notifier_dict.values())[0]
        
        return notifier
    
    def get_claims_date(self, document: str) -> str:
        """
        Extracts the date of the insurance claim from the document using a
        structured prompt.

        This method formats a detailed prompt that instructs the language
        model to read through the provided insurance claim document and
        return the date of the claim in a JSON format. The prompt emphasizes
        careful reading and structured JSON response that includes both the
        reasoning process ("Thinking") and the identified date ("Date").
        
        """
        instructions = """
            Ihre Aufgabe besteht darin, den Text aus dem Schadensbericht zu 
            lesen und einen JSON mit Ihrem Denkprozess und dem Schadensdatum 
            zurückzusenden. Befolgen Sie die Anweisungen und Beispiele, um auf 
            die gleiche Weise Antworten zu geben.

            <instructions>
            - Beginnen Sie Ihre Antwort immer mit '{{'.
            - Die Antwort darf keine einleitenden Sätze enthalten, geben Sie 
                ausschließlich ein JSON zurück.
            - Das JSON soll die keys 'Thinking' und 'Date' enthalten.
            - Geben Sie Datum unter dem key 'Date' im Format TT.MM.JJJJ zurück.
            </instructions>

            <example_1>
            "120552.78005.076-00; Schadenmeldung Vetragsnr.: 5432862421
            Eve Tokes An. NKO Importer PROD.de 08.04.2024 09:30
            Von: "Eve Tokes" <eve@gmail.com>
            An: NKO Importer schaden@gmail.com
            Bitte Antwort an info@gmail.com
            
            Eingangszeitpunkt: 08.04.2024 09:29:49
            Sehr geehrte Damen und Herren,
            unser gemeinsamer Kunde Herr Gendel Falken informierte mich über einen
            Fahrraddiebstahl. Entsprechende Unterlagen, siehe Anlage vom Kunden.
            Ich bitte um Schadenbearbeitung.
            Mit freundlichen Grüßen
            Eve Tokes
                        
            {{"Thinking": "Es gab eine Information über den Fahrraddiebstahl
            von Herrn Sven Albrechtsen, er gab jedoch nicht an, wann es passierte.
            Es gibt also kein Datum, wann der Diebstahl stattgefunden hat.",
            "Date": "None"}}
            </example_1>

            <example_2> 
            "240122.65668.079-00;2245630646 / Gundula Fisch / Schadenmeldung
            Sturmschaden\nDavid Otter An: NKO_Importer_PROD.de 
            22.03.2024 17:15\nVon: ""David Otter"" <fake@fakewebsite.com>
            fake@fakewebsite.com\nBitte Antwort an fake@fakewebsite.com\nAN: 
            fake@fakewebsite.com\nEingangszeitpunkt: 22.03.2024 17:14:12
            Sehr geehrte Damen und Herren,\nhiermit melden wir Ihnen die weiter unten
            von der VN näher beschriebenen Schaden\nmit der Bitte um Regulierung.
            Beim Schadentag muss es sich um den fakewebsite.com oder 
            fakewebsite.com März gehandelt haben.\nMit freundliche Grüßen\n441111m.1. 
            1111111.11*\nSIMPL\nDavid Otter\nSachbearbeiter\nfake@fakewebsite.com
            fakewebsite.com\nTelefon: 0000/0000000Fax: 0000/0000000Büroanschrift:
            Leimberg 123, 54231 Stolberg\nRegistrierungsnummer D-UBK3-BQZH5-31;
            \nVon: fake@fakewebsite.com <fake@fakewebsite.com>\nGesendet:
            Montag, 22. März 2024 16:25\nAn: David Otter <fake@fakewebsite.com>\n
            Betreff: Sturmschaden\nGuten Tag Herr Otter!\nWie heute morgen bereits
            mit Frau Schmidt besprochen, erhalten Sie anbei die\nSchadensmeldung zum
            Sturmschaden kurz vor Ostern.\nDer Regenhut vom Edelstahlschornstein ist
            ?weggeweht"" worden.\nDas Angebot vom Dachdecker sowie Fotos mit und ohne
            Regenhut sind beigefügt.\nDer genaue Tag kann leider nicht bestimmt
            werden, da wir den Schaden erst später\nbemerkt haben.\nDer Regenhut
            wurde tatsächlich weggeweht und ist nicht mehr auffindbar;
            glücklicherweise wurde niemand verletzt.\nWir möchten Sie bitten, alles
            Notwendige zur Regulierung in die Wege zu leiten.\nBei Rückfragen gerne
            melden unter 0173/1286423\nVielen Dank vorab!\nFreundliche Grüsse\n
            Gundula Fisch.
            
            {{"Thinking": "Das Dokument beschreibt einen durch Sturm
            verursachten Schaden. Das angenommene Datum des Sturms liegt zwischen
            dem 21.03.2024 oder dem 22.03.2024. Gibt es kein konkretes Datum. Bei
            aufeinanderfolgenden Daten zum Vorfall wählen Sie immer das erste Datum,
            also den 21.03.2024.", "Date": "21.03.2024"}}
            </example_2>
            
            """
            
        messages=[
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": document},
            {"role": "user", "content": instructions}
        ]
        
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                response_format={ "type": "json_object" },
                messages=messages
            )
        
        date_dict = json.loads(response.choices[0].message.content)
        date = list(date_dict.values())[1]
        
        return date
        
    def get_claims_objekt(self, document: str) -> str:
        """
        Identifies the category of the object involved in the insurance claim
        based on the document content.

        This method uses a predefined template to generate a prompt that
        instructs the language model
        to categorize the damage or loss reported in the insurance claim document.
        The model is expected to respond with an abbreviation representing the
        best-fitting category among predefined options.
        """
        
        instructions = """
                <instructions>
                - Beantworte, unter Verwendung der entsprechenden Kategorie, 
                welche Schaden-Kategorie am besten passt. Kategorien: 
                [Glass, Hausrat, Wohngebäude, Kasko, Kraftfahrthaftpflicht, Sonstige]
                - Beantworte die Frage in dem Du ausschließlich das Kürzel der 
                jeweiligen Kategorie ohne weitere Leerzeichen oder Sonderzeichen 
                zurückgibst:  Kürzel: Glass = 'GL', Hausrat = 'HR', 
                Wohngebäude = 'WG', Kasko = 'KF', Kraftfahrthaftpflicht = 'KH', 
                Sonstige = 'Other'
                - Hinweis: Ein Fahrraddiebstahl fällt in der Regel in die Kategorie 
                'Hausrat'
                </instructions>
                """
                
        messages=[
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": document},
            {"role": "user", "content": instructions}
        ]
        
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                response_format={ "type": "json_object" },
                messages=messages
            )
        
        claims_object_dict = json.loads(response.choices[0].message.content)
        claims_object = list(claims_object_dict.values())[0]
        
        return claims_object
    
    def cause_mapper(
        self, claims_object: str, claims_type: str, cause_type: str
    ) -> str | None:

        logger.info(f"Schaden Objekt: {claims_object}")
        logger.info(f"Schaden Typ: {claims_type}")

        if (claims_object, claims_type) in self.special_cases:
            urs_art = self.special_cases[(claims_object, claims_type)]

            return str(urs_art)
        if claims_object in ["WG", "HR"] and claims_type == "GL":
            logger.info(f"Schaden-Ursachen-Art: {0}")
            return "0"

        try:
            logger.info(
                f"Schaden-Ursachen-Art: {str(self.cause_mapping[claims_type][cause_type])}"
            )

            urs_art = str(self.cause_mapping[claims_type][cause_type])

            return urs_art
        except KeyError as e:
            logger.warning(f"Invalid {e.args[0]}")
            return None
        
    def fde_first_notification_of_loss(self, document: str) -> dict:
        """
        Executes all related methods to process the first notification of loss
        based on the given document.

        This function orchestrates the execution of several methods to extract
        comprehensive information
        about an insurance claim from a document. It first determines the claims
        type and, based on that,
        it may conditionally proceed to determine the cause type and other
        details.
        """
        
        try:
            claims_type = self.get_claims_type(document)
            if isinstance(claims_type, str):
                claims_type = claims_type.strip()
            else:
                claims_type = None
        except ValueError:
            claims_type = None

        if desc := self.get_cause(document=document, claims_type=claims_type):
            cause_description = desc.strip()
        else:
            cause_description = None

        claims_object = self.get_claims_objekt(document).strip()
        cause_alphanumerical = self.cause_mapper(
            claims_object=claims_object,
            claims_type=claims_type,
            cause_type=cause_description,
        )

        result = {
            "type": claims_type,
            "cause": cause_alphanumerical,
            "notifier": self.get_notifier(document).strip(),
            "date": self.get_claims_date(document),
            "objekt": claims_object,
        }

        return result
    
@dataclass   
class FirstNotificationOfLoss:
    """
    Represents the first notification of a loss claim in insurance
    processing.

    This class is designed to hold all necessary details about an individual
    insurance claim as reported initially. This includes identifiers and
    descriptions of the damage and the cause and date of the
    damage.

    Attributes:
        doc_id (str): The document identifier unique to this claim.
        schaden_objekt (str): Description of the object or property damaged.
        schaden_typ (str): The type of damage incurred.
        sd_urs_art (str): The type of cause that led to the damage.
        schaden_datum (str): The date on which the damage occurred, formatted as a string.
    """

    doc_id: str
    schaden_objekt: str
    schaden_typ: str
    sd_urs_art: str
    schaden_datum: str


    def group_sd_urs_art(df: pd.DataFrame) -> pd.DataFrame:
        """Group sd_urs_art into main categories based on sd_typ_kennung.

        Args:
            df: pd.DataFrame to be grouped

        Returns:
            DataFrame containing main categories for sd_urs_art based on sd_typ_kennung
        """
        logger.info("Grouping sd_urs_art and schaden_objekt")
        list_sub_dfs = [
            y
            for x, y in df.groupby(
                ["sd_typ_kennung", "schaden_objekt"], as_index=False
            )
        ]

        result_df_list = []

        for sub_df in list_sub_dfs:
            if sub_df["sd_typ_kennung"].iloc[0] == "LW":
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                sub_df.sd_urs_art = sub_df.sd_urs_art.apply(lambda x: x[0])
                sub_df = sub_df.replace(
                    {"sd_urs_art": {r"^[^0249]*$": "0"}}, regex=True
                )
                result_df_list.append(sub_df)

            elif sub_df["sd_typ_kennung"].iloc[0] == "ST":
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                sub_df["sd_urs_art"] = (
                    sub_df["sd_urs_art"]
                    .apply(lambda x: x[0] if x not in ["10", "19"] else x)
                    .apply(
                        lambda x: x
                        if x in ["0", "1", "10", "19", "4", "7"]
                        else "1"
                    )
                    .apply(lambda x: "10" if x == "1" else x)
                )
                result_df_list.append(sub_df)

            elif sub_df["sd_typ_kennung"].iloc[0] == "EL":
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                sub_df.sd_urs_art = sub_df.sd_urs_art.apply(lambda x: x[0])
                sub_df = sub_df.replace(
                    {"sd_urs_art": {r"^.*[^2]$": "0"}}, regex=True, inplace=False
                )
                result_df_list.append(sub_df)

            elif sub_df["sd_typ_kennung"].iloc[0] == "ED":
                sub_df["sd_urs_art"] = (
                    sub_df["sd_urs_art"]
                    .astype(str)
                    .apply(lambda x: x[0] if x not in ["70", "79"] else x)
                    .apply(lambda x: x if x in ["0", "7", "70", "79"] else "0")
                    .apply(lambda x: "70" if x == "7" else x)
                )

                result_df_list.append(sub_df)

            elif (
                sub_df["sd_typ_kennung"].iloc[0] == "GL"
                and sub_df["schaden_objekt"].iloc[0] == "GL"
            ):
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                sub_df["sd_urs_art"] = sub_df["sd_urs_art"].replace("9", "09")
                sub_df["sd_urs_art"] = sub_df["sd_urs_art"].replace(
                    {"^(?!09$|11$|12$).*$": "00"}, regex=True, inplace=False
                )
                result_df_list.append(sub_df)

            elif sub_df["sd_typ_kennung"].iloc[0] == "GL" and (
                sub_df["schaden_objekt"].iloc[0] in ["HR", "WG"]
            ):
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                sub_df["sd_urs_art"] = "00"
                result_df_list.append(sub_df)

            elif sub_df["sd_typ_kennung"].iloc[0] == "FE":
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                sub_df.sd_urs_art = sub_df.sd_urs_art.apply(lambda x: x[0]).apply(
                    lambda x: x if x in ["6", "9"] else "0"
                )

                result_df_list.append(sub_df)

            elif (
                sub_df["sd_typ_kennung"].iloc[0] == "VK"
                or sub_df["sd_typ_kennung"].iloc[0] == "TK"
            ):
                sd_typ_kennung = sub_df["sd_typ_kennung"].iloc[0]
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(int)
                categories = {
                    "VK": {1: [51], 2: [562], 3: [564, 561, 57, 563, 565]},
                    "TK": {
                        77: [77],
                        741: [741, 742, 743, 744],
                        782: [782],
                        751: [751],
                        733: [
                            71,
                            78,
                            781,
                            72,
                            79,
                            791,
                            76,
                            753,
                            731,
                            732,
                            752,
                            733,
                            734,
                            771,
                            783,
                            792,
                            793,
                        ],
                    },
                }

                categories_mapping = {
                    **{
                        "VK": {
                            val: k
                            for k, l in categories["VK"].items()
                            for val in l
                        }
                    },
                    **{
                        "TK": {
                            val: k
                            for k, l in categories["TK"].items()
                            for val in l
                        }
                    },
                }

                sub_df["sd_urs_art"] = sub_df["sd_urs_art"].map(
                    categories_mapping[sd_typ_kennung]
                )
                sub_df.sd_urs_art = sub_df.sd_urs_art.astype(str)
                result_df_list.append(sub_df)
            else:
                result_df_list.append(sub_df)

        result_df = pd.concat(result_df_list)
        result_df = result_df.reset_index(drop=True)

        return result_df
        
@dataclass
class DirksClaims:
    """
    A class representing a collection of claims data for Dirk's insurance claims
     processing.

    Attributes:
        doc_id_list (List[str]): List of document identifiers for the claims.
        schaden_objekt_list (List[str]): List of objects involved in the claims.
        schaden_typ_list (List[str]): List of types of damage involved in the
        claims.
        sd_urs_art_list (List[str]): List of cause types for the damages.
        schaden_datum_list (List[str]): List of dates when the damages occurred.

    The class aggregates information from individual lists into structured
    claims data using the `FirstNotificationOfLoss` class for each claim entry.
    """

    doc_id_list: List[str]
    schaden_objekt_list: List[str]
    schaden_typ_list: List[str]
    sd_urs_art_list: List[str]
    schaden_datum_list: List[str]
    
    def __post_init__(self):
        """
        Post-initialization processing to convert lists of claims data into a
        list of `FirstNotificationOfLoss` objects.

        This method automatically runs after the class is instantiated to bundle
        the provided claims lists into structured objects. It iterates over the
        combined entries of all provided lists and creates a
        `FirstNotificationOfLoss` object for each set of entries.

        Raises:
            ValueError: If the lengths of the input lists do not match,
                indicating inconsistent data which cannot be correctly paired
                into claims.
        """
        if not all(
            len(lst) == len(self.doc_id_list)
            for lst in [
                self.schaden_objekt_list,
                self.schaden_typ_list,
                self.sd_urs_art_list,
                self.schaden_datum_list,
            ]
        ):
            raise ValueError("All input lists must have the same length.")

        self.info = [
            FirstNotificationOfLoss(
                doc_id=doc_id,
                schaden_objekt=schaden_objekt,
                schaden_typ=schaden_typ,
                sd_urs_art=sd_urs_art,
                schaden_datum=schaden_datum,
            )
            for doc_id, schaden_objekt, schaden_typ, sd_urs_art, schaden_datum in zip(
                self.doc_id_list,
                self.schaden_objekt_list,
                self.schaden_typ_list,
                self.sd_urs_art_list,
                self.schaden_datum_list,
            )
        ]
               
import dataclasses
import enum
import multiprocessing
import os
import sys
import time

import click
import requests
import spacy
import tqdm
from bs4 import BeautifulSoup


class Gender(enum.Enum):
    ALL = "all"
    FEMALE = "F"
    MALE = "M"
    UNKNOWN = "U"


class Age(enum.Enum):
    ALL = "all"
    UNKNOWN = "0"
    AGE_17_23 = "1"
    AGE_24_30 = "2"
    AGE_31_50 = "3"
    AGE_OVER_51 = "4"


class Role(enum.Enum):
    ALL = "all"
    JUNIOR_FACULTY = "JF"
    JUNIOR_GRADUATE_STUDENT = "JG"
    JUNIOR_UNDERGRADUATE = "JU"
    POST_DOCTORAL_FELLOW = "PD"
    RESEARCHER = "RE"
    SENIOR_FACULTY = "SF"
    SENIOR_GRADUATE_STUDENT = "SG"
    STAFF = "ST"
    SENIOR_UNDERGRADUATE = "SU"
    UNKNOWN = "UN"
    VISTOR_OTHER = "VO"


class NativeSpeakerStatus(enum.Enum):
    ALL = "all"
    NON_NATIVE_SPEAKER = "NNS"
    NEAR_NATIVE_SPEAKER = "NRN"
    NATIVE_SPEAKER_AMERICAN_ENGLISH = "NS"
    NATIVE_SPEAKER_OTHER_ENGLISH = "NSO"
    UNKNOWN = "UN"


class FirstLang(enum.Enum):
    ALL = "all"
    ARABIC = "ARA"
    ARMENIAN = "ARM"
    CANTONESE = "CAN"
    MANDARIN = "CHI"
    CZECH = "CZE"
    DUTCH = "DUT"
    ESTONIAN = "EST"
    FARSI = "FAR"
    FRENCH = "FRE"
    GERMAN = "GER"
    GUJARATI = "GUJ"
    HEBREW = "HEB"
    HINDI = "HIN"
    HUNGARIAN = "HUN"
    IBO = "IBO"
    INDONESIAN = "IND"
    ITALIAN = "ITA"
    JAPANESE = "JAP"
    KANNADA = "KAN"
    KOREAN = "KOR"
    MACEDONIAN = "MAC"
    MARATHI = "MAR"
    POLISH = "POL"
    PORTUGUESE = "POR"
    RUSSIAN = "RUS"
    SOUTH_AFRICAN_ENGLISH = "SAE"
    SLOVAK = "SLO"
    SPANISH = "SPA"
    SWAHILI = "SWA"
    SWEDISH = "SWE"
    TAGALOG = "TAG"
    TELEGU = "TEL"
    THAI = "THA"
    TURKISH = "TUR"
    BRITISH_ENGLISH = "UKE"
    UKRAINIAN = "UKR"
    UNKNOWN = "UNK"
    URDU = "URD"
    VIETNAMESE = "VIE"


class SpeechEventType(enum.Enum):
    ALL = "all"
    ADVISING_SESSION = "ADV"
    COLLOQUIUM = "COL"
    DISSERTATION_DEFENSE = "DEF"
    DISCUSSION_SECTIONS = "DIS"
    INTERVIEW = "INT"
    LAB_SECTION = "LAB"
    LECTURE_LARGE = "LEL"
    LECTURE_SMALL = "LES"
    MEETING = "MTG"
    OFFICE_HOURS = "OFC"
    SEMINAR = "SEM"
    STUDY_GROUP = "SGR"
    STUDENT_PRESENTATIONS = "STP"
    SERVICE_ENCOUNTER = "SVC"
    TOUR = "TOU"


class AcademicDivision(enum.Enum):
    ALL = "all"
    BIOLOGICAL_AND_HEALTH_SCIENCES = "BS"
    HUMANITIES_AND_ARTS = "HA"
    NOT_APPLICABLE_OTHER = "NA"
    PHYSICAL_SCIENCES_AND_ENGINEERING = "PS"
    SOCIAL_SCIENCES_AND_EDUCATION = "SS"


class AcademicDiscipline(enum.Enum):
    ALL = "all"
    AFROAMERICAN_AND_AFRICAN_STUDIES = "095"
    AMERICAN_CULTURE = "105"
    ANTHROPOLOGY = "115"
    ARCHITECTURE = "125"
    ASIAN_LANGUAGES_AND_CULTURES = "140"
    ASTRONOMY = "150"
    BIOMEDICAL_ENGINEERING = "165"
    BIOLOGY = "175"
    BUSINESS_ADMINISTRATION = "185"
    CHEMICAL_ENGINEERING = "195"
    CHEMISTRY = "200"
    CIVIL_AND_ENVIRONMENTAL_ENGINEERING = "205"
    CLASSICAL_STUDIES = "215"
    COMMUNICATION = "220"
    COMPUTER_SCIENCE_GENERAL_UNDERGRADUATE = "235"
    ELECTRICAL_ENGIN_AND_COMPUTER_SCIENCE = "270"
    ECONOMICS = "280"
    EDUCATION = "285"
    ENGINEERING_GENERAL_UNDERGRADUATE = "295"
    ENGLISH = "300"
    ENGLISH_COMPOSITION = "301"
    GEOLOGICAL_SCIENCES = "305"
    HISTORY = "315"
    HISTORY_OF_ART = "320"
    INDUSTRIAL_AND_OPERATIONS_ENGINEERING = "330"
    INFORMATION_AND_LIBRARY_SCIENCE = "335"
    PUBLIC_POLICY = "340"
    LINGUISTICS = "355"
    MECHANICAL_ENGINEERING = "365"
    MATH = "385"
    IMMUNOLOGY = "400"
    MICROBIOLOGY = "405"
    FINE_ARTS = "420"
    NATURAL_RESOURCES = "425"
    NUCLEAR_ENGINEERING = "445"
    NURSING = "450"
    PHILOSOPHY = "475"
    PHYSICS = "485"
    POLITICAL_SCIENCE = "495"
    PSYCHOLOGY = "500"
    RELIGION = "542"
    ROMANCE_LANGUAGES_AND_LITERATURE = "545"
    SOCIAL_WORK = "560"
    SOCIOLOGY = "565"
    STATISTICS = "575"
    TECHNICAL_COMMUNICATIONS = "578"
    WOMENS_STUDIES = "605"
    ADVISING = "700"
    MISC_NONDEPARTMENTAL = "999"


class ParticipantLevel(enum.Enum):
    ALL = "all"
    JUNIOR_FACULTY = "JF"
    JUNIOR_GRADUATE_STUDENTS = "JG"
    JUNIOR_UNDERGRADUATES = "JU"
    MIXED_FACULTY = "MF"
    MIXED_GRADUATE_STUDENTS = "MG"
    MIXED_UNDERGRADUATES = "MU"
    MIXED_STUDENTS_FAC_STAFF = "MX"
    SENIOR_FACULTY = "SF"
    SENIOR_GRADUATE_STUDENTS = "SG"
    STAFF = "ST"
    SENIOR_UNDERGRADUATES = "SU"
    VISITOR_OTHER = "VO"


class InteractivityRating(enum.Enum):
    ALL = "all"
    HIGHLY_INTERACTIVE = "HI"
    HIGHLY_MONOLOGIC = "HM"
    MOSTLY_INTERACTIVE = "MI"
    MOSTLY_MONOLOGIC = "MM"
    MIXED = "MX"


@dataclasses.dataclass
class QueryArguments:
    query: str
    gender: Gender = Gender.ALL
    age: Age = Age.ALL
    role: Role = Role.ALL
    nss: NativeSpeakerStatus = NativeSpeakerStatus.ALL
    first_lang: FirstLang = FirstLang.ALL
    speech_event_type: SpeechEventType = SpeechEventType.ALL
    acad_div: AcademicDivision = AcademicDivision.ALL
    acad_disc: AcademicDiscipline = AcademicDiscipline.ALL
    part_level: ParticipantLevel = ParticipantLevel.ALL
    disc_mode: InteractivityRating = InteractivityRating.ALL


class Corpus:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.data = {}

    def form_url(self, args: QueryArguments):
        url = f"""
            https://quod.lib.umich.edu/cgi/c/corpus/corpus?
            c=micase;
            cc=micase; type=simple;
            q1={args.query.replace(" ", "+")};
            gender={args.gender.value};
            age={args.age.value};
            role={args.role.value};
            nss={args.nss.value};
            firstlang={args.first_lang.value};
            speecheventtype={args.speech_event_type.value};
            acaddiv={args.acad_div.value};
            acaddisc={args.acad_disc.value};
            partlevel={args.part_level.value};
            discmode={args.disc_mode.value};
            view=reslist;
            subview=short;
            sort=occur;
            start=1;
            size=25;
            confirm=1
        """
        return url.replace("\n", "").replace(" ", "")

    def get_urls(self, url: str) -> list[str]:
        response = requests.get(url).text
        if "Your usage has exceeded normal limits" in response:
            print("\nYo, chill! Take it easy!")
            print("Usage is through the roof. Try again in a few minutes.")
            sys.exit()
        html = BeautifulSoup(response, features="html.parser")
        return [l["href"] for l in html.body.find_all("a") if l.getText() == "view"]

    def get_text(self, L, url: str) -> str:
        html = BeautifulSoup(requests.get(url).text, features="html.parser")
        text = html.body.find("div", {"id": "contextResult"}).text
        while "  " in text:
            text = text.replace("  ", " ")
        L.append(text)

    def get_texts(self, urls: list[str], processes: int = 64) -> list[str]:
        texts = []
        total = len(urls)
        pbar = tqdm.tqdm(total=total)

        for url in urls:
            self.get_text(texts, url)
            pbar.update(1)
            time.sleep(1)

        """ Async magic fails
        for chunk in [urls[i : i + processes] for i in range(0, total, processes)]:
            L = []
            with multiprocessing.Manager() as manager:
                L = manager.list()
                processes = []
                for url in chunk:
                    p = multiprocessing.Process(target=self.get_text, args=(L, url))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                    pbar.update(1)

                L = list(L)
            texts += L
        """

        pbar.close()
        return texts

    def text_to_sentences_with_keyword(self, text: str, keyword: str) -> list[str]:
        return [
            sentence.text
            for sentence in self.nlp(text).sents
            if keyword in sentence.text
        ]

    def get_data(self, urls: list[str], query: str) -> None:
        texts = self.get_texts(urls)
        for url, text in zip(urls, texts):
            if "Speaker Information Restricted" in text:
                continue
            sentences = self.text_to_sentences_with_keyword(text, query)
            for sentence in sentences:
                self.data[url] = {"entry": text, "sentence": sentence}

    def remove_dups(self) -> int:
        unique = []
        to_remove = []
        for k, v in self.data.items():
            if v["sentence"] in unique:
                to_remove.append(k)
            unique.append(v["sentence"])
        for k in to_remove:
            del self.data[k]
        return len(to_remove)

    def download(self, args: QueryArguments, path: str) -> None:
        url = self.form_url(args)

        print(f"Querying...\n{url}")

        urls = self.get_urls(url)

        if len(urls) == 0:
            print("\nCan't find any entries with those query arguments!")
            return

        urls = urls[:50]

        print(f"\nFound {len(urls)} entries!")
        cont = click.confirm(
            "Would you like to download all of them?",
            default=False,
            show_default=True,
        )
        if not cont:
            print("\nAborted!")
            return

        self.get_data(urls, args.query)

        print("\nProcessing...")
        print(
            f'- Removed {len(urls) - len(self.data)} entries containing "Speaking'
            ' Information Restricted"'
        )
        dups = self.remove_dups()
        print(f"- Removed {dups} duplicates")
        print(f"Downloaded {len(self.data)} entries!")

        print(f"\nWriting to file {path}...")

        if os.path.exists(path):
            print(f"File {path} already exists.")
            cont = click.confirm(
                f"Would you like to overwrite the file?",
                default=False,
                show_default=True,
            )
            if not cont:
                print("\nAborted!")
                return

        with open(path, "w") as file:
            file.write("id,url,entry,sentence\n")
            for i, item in enumerate(self.data.items(), start=1):
                url, vals = item
                file.write(
                    f"{i},\"{url}\",\"{vals['entry']}\",\"{vals['sentence']}\"\n"
                )

        print("\nDone!")


if __name__ == "__main__":
    # args
    output_file = "results.csv"
    args = QueryArguments(
        query="will",
        # query="going to",
        gender=Gender.ALL,
        age=Age.ALL,
        role=Role.ALL,
        nss=NativeSpeakerStatus.NATIVE_SPEAKER_AMERICAN_ENGLISH,
        first_lang=FirstLang.ALL,
        speech_event_type=SpeechEventType.LECTURE_SMALL,
        acad_div=AcademicDivision.ALL,
        acad_disc=AcademicDiscipline.ALL,
        part_level=ParticipantLevel.ALL,
        disc_mode=InteractivityRating.ALL,
    )

    Corpus().download(args, output_file)

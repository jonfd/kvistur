#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

patch = {
    "ABC-Waffen": "Waffen",
    "Abstammungsverhältnis": "verhältnis",
    "Aggressionspotenzial": "potenzial",
    "Agrarsubvention": "subvention",
    "Allerheiligen": "heiligen",
    "Anführungsstrich": "strich",
    "Arbeitsverhältnisse": "verhältnisse",
    "Atembeschwerden": "beschwerden",
    "Augustiner-Chorherren-Stift": "Chorherren-Stift",
    "Basisdaten": "daten",
    "Betriebsdaten": "daten",
    "Bilddaten": "daten",
    "Biogeografie": "geografie",
    "Eckdaten": "daten",
    "Eingusstrichter": "trichter",
    "Einsparpotenzial": "potenzial",
    "E-Marketingprojekt": "projekt",
    "Erfolgspotenzial": "potenzial",
    "Fahreigenschaft": "eigenschaft",
    "Festtagskleidung": "kleidung",
    "Fischgründe": "gründe",
    "Fliegerkräfte": "kräfte",
    "Füllsand": "sand",
    "Ganztagsbetreuung": "betreuung",
    "Gerichtsvollzieherin": "vollzieherin",
    "Getreideerzeugnis": "erzeugnis",
    "Glasschmelze": "schmelze",
    "Grenzfestlegung": "festlegung",
    "Hämolymphe": "lymphe",
    "Hauptstadtklub": "klub",
    "Himbeer": "beer",
    "Hochtouren": "touren",
    "Hofbeamtin": "beamtin",
    "Hydrodynamik": "dynamik",
    "Hydroxid": "oxid",
    "Internetprotokoll": "protokoll",
    "Jobbörse": "börse",
    "Jugendbande": "bande",
    "Jungferninseln": "inseln",
    "Kapazitätsgründe": "gründe",
    "Kenndaten": "daten",
    "Konjunkturdaten": "daten",
    "Kontaktdaten": "daten",
    "Kostendaten": "daten",
    "Kronjuwel": "juwel",
    "Kronland": "land",
    "Kundendaten": "daten",
    "Kurzweil": "weil",
    "Küstenstädtchen": "städtchen",
    "Laborschuh": "schuh",
    "Langstreckenläufer": "läufer",
    "Lebensdaten": "daten",
    "Lebenserinnerungen": "erinnerungen",
    "Lebensgewohnheiten": "gewohnheiten",
    "Lebensverhältnisse": "verhältnisse",
    "Lehranalyse": "analyse",
    "Leistungsdaten": "daten",
    "Lungenstiche": "stiche",
    "Magenbeschwerden": "beschwerden",
    "Magen-Darm-Beschwerden": "beschwerden",
    "Medaillenaussichten": "aussichten",
    "Messdaten": "daten",
    "Metallschmelze": "schmelze",
    "Mittelgebirgszug": "zug",
    "Münzmeister": "meister",
    "Münzprägestätte": "prägestätte",
    "Muttergottes": "gottes",
    "Naturschutzgründe": "gründe",
    "Nilosaharanisch": "saharanisch",
    "Nordskandinavien": "skandinavien",
    "Nutzdaten": "daten",
    "Personendaten": "daten",
    "Pflanzensamen": "samen",
    "Polizeivollzugsbeamtin": "vollzugsbeamtin",
    "Priesterbruderschaft": "bruderschaft",
    "Produktionspotenzial": "potenzial",
    "Rahmendaten": "daten",
    "Reichsinsignien": "insignien",
    "Rohdaten": "daten",
    "Satzkonstituente": "konstituente",
    "Schwarzdrossel": "drossel",
    "Scrollbalken": "balken",
    "Segeljacht": "jacht",
    "Servicepersonal": "personal",
    "Sichtschneise": "schneise",
    "Skiklub": "klub",
    "Sonnenwendwolfsmilch": "wolfsmilch",
    "Stadtinneres": "inneres",
    "Standesbeamtin": "beamtin",
    "Stoppball": "ball",
    "Straßenverhältnisse": "verhältnisse",
    "Streitkräfte": "kräfte",
    "Südmarokko": "marokko",
    "Textdaten": "daten",
    "Tiefkühlprodukt": "produkt",
    "Tierlaut": "laut",
    "Trennschleifer": "schleifer",
    "UCI-Straßen-Weltmeisterschaften": "Straßen-Weltmeisterschaften",
    "Umweltdaten": "daten",
    "Urkundsbeamter": "beamter",
    "Vorgängerversion": "version",
    "Wechseljahre": "jahre",
    "Wetterbedingungen": "bedingungen",
    "Wetterdaten": "daten",
    "Winkelzüge": "züge",
    "Wirtschaftsdaten": "daten",
    "Zahlungsbedingungen": "bedingungen",
    "Zwergwolfsmilch": "wolfsmilch",
    "Blütenhüllblatt": "blatt"
}


def read_germanet(path):
    with open(path, encoding='utf-8') as f:
        lines = [line.split('\t') for line in f.read().splitlines()][2:]

    heads = {}

    for compound, modifier, head in lines:
        # Only take the last word from multiword expressions, e.g., "periphere arterielle Verschlusskrankheit"
        compound = compound.split()[-1].strip()
        head = head.strip()

        # Skip base words
        if not modifier or not head:
            continue

        heads[compound] = head

    return heads


def main():
    germanet_path = sys.argv[1]
    out_path = sys.argv[2]

    # Read GermaNet 12.0
    germanet = read_germanet(germanet_path)

    # Some compounds are inflected but their head is lemmatized, e.g., "Trennschleifer" with the head "Schleife". Since
    # we're determining the position of the binary split based on the length of the head, we'll update those heads.
    # Also, we correct a very small number of errors, such as "Stoppball" which is listed with the head "Pferd".
    germanet.update(patch)

    # Write output
    output = {compound.lower(): head.lower() for compound, head in germanet.items()}
    print("{:,} word forms processed".format(len(germanet)))

    with open(out_path, 'w', encoding='utf-8') as f:
        for compound, head in output.items():
            assert compound.endswith(head)
            mod = compound[:-len(head)]
            f.write(f"{mod}_{head}\n")


if __name__ == '__main__':
    main()

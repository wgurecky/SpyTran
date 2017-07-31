import periodictable as pt
import json


def genIsoDict():
    matLib = {}
    for el in pt.elements:
        for iso in el:
            if el.number < 100:
                zaid = str(el.number) + str('{0:03d}'.format(iso.isotope))
                matLib[str(iso.element).lower() + str(iso.isotope)] = {
                    'M': iso.mass,
                    'A': int(el.number),
                    'zaid': int(zaid)
                }
    return matLib


if __name__ == "__main__":
    matLib = genIsoDict()
    with open('isoDict.json', 'w') as outfile:
        json.dump(matLib, outfile, sort_keys=True, indent=4, separators=(', ', ': '))

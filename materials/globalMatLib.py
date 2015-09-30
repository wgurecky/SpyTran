#!/usr/bin/python

matLib = {}
matLib['U234'] = {'M': 234.0409456,  # Atomic mass [kg/mol]
                  'A': 92.0,         # Number of protons
                  'zaid': '92234'}   # ZAID identifier
matLib['U235'] = {'M': 235.043931,
                  'A': 92.0,
                  'zaid': '92235'}
matLib['U236'] = {'M': 236.0455619,
                  'A': 92.0,
                  'zaid': '92236'}
matLib['U238'] = {'M': 238.0507826,
                  'A': 92.0,
                  'zaid': '92238'}
matLib['H1'] = {'M': 1.007825,
                'A': 1.0,
                'zaid': '1001'}
matLib['O16'] = {'M': 15.9949146,
                 'A': 8.0,
                 'zaid': '8016'}
matLib['B10'] = {'M': 10.012937,
                 'A': 5.0,
                 'zaid': '5010'}
matLib['B11'] = {'M': 11.0093055,
                 'A': 5.0,
                 'zaid': '5011'}
matLib['C12'] = {'M': 12.0,
                 'A': 6,
                 'zaid': 6000
                 }
matLib['Si28'] = {'M': 27.9769265,
                  'A': 14,
                  'zaid': 14028
                  }
matLib['Si29'] = {'M': 28.9764947,
                  'A': 14,
                  'zaid': 14029
                  }
matLib['Si30'] = {'M': 29.9737702,
                  'A': 14,
                  'zaid': 14030
                  }
matLib['P31'] = {'M': 30.9737615,
                 'A': 15,
                 'zaid': 15031
                 }
matLib['Cr50'] = {'M': 49.9460496,
                  'A': 24,
                  'zaid': 24050
                  }
matLib['Cr52'] = {'M': 51.9405119,
                  'A': 24,
                  'zaid': 24052
                  }
matLib['Cr53'] = {'M': 52.9406538,
                  'A': 24,
                  'zaid': 24053
                  }
matLib['Cr54'] = {'M': 53.93888,
                  'A': 24,
                  'zaid': 24054
                  }
matLib['Mn55'] = {'M': 54.938049,
                  'A': 25,
                  'zaid': 25055
                  }
matLib['Fe54'] = {'M': 53.93961,
                  'A': 26,
                  'zaid': 26054
                  }
matLib['Fe56'] = {'M': 55.93494,
                  'A': 26,
                  'zaid': 26056
                  }
matLib['Fe57'] = {'M': 56.9353987,
                  'A': 26,
                  'zaid': 26057
                  }
matLib['Fe58'] = {'M': 57.93328,
                  'A': 26,
                  'zaid': 26058
                  }
matLib['Ni58'] = {'M': 57.9353,
                  'A': 28,
                  'zaid': 28058
                  }
matLib['Ni60'] = {'M': 59.9307,
                  'A': 28,
                  'zaid': 28060
                  }
matLib['Ni61'] = {'M': 60.93106,
                  'A': 28,
                  'zaid': 28061
                  }
matLib['Ni62'] = {'M': 61.9283488,
                  'A': 28,
                  'zaid': 28062
                  }
matLib['Ni64'] = {'M': 63.9279696,
                  'A': 28,
                  'zaid': 28064
                  }
matLib['Zr90'] = {'M': 89.904,
                  'A': 40,
                  'zaid': 40090
                  }
matLib['Zr91'] = {'M': 90.9056,
                  'A': 40,
                  'zaid': 40091
                  }
matLib['Zr92'] = {'M': 91.905,
                  'A': 40,
                  'zaid': 40092
                  }
matLib['Zr94'] = {'M': 93.9063168,
                  'A': 40,
                  'zaid': 40094
                  }
matLib['Zr96'] = {'M': 95.90827,
                  'A': 40,
                  'zaid': 40096
                  }
matLib['Sn112'] = {'M': 111.90482,
                   'A': 50,
                   'zaid': 50112
                   }
matLib['Sn114'] = {'M': 113.90278,
                   'A': 50,
                   'zaid': 50114
                   }
matLib['Sn115'] = {'M': 114.90334,
                   'A': 50,
                   'zaid': 50115
                   }
matLib['Sn116'] = {'M': 115.9017441,
                   'A': 50,
                   'zaid': 50116
                   }
matLib['Sn117'] = {'M': 116.9029538,
                   'A': 50,
                   'zaid': 50117
                   }
matLib['Sn118'] = {'M': 117.9016,
                   'A': 50,
                   'zaid': 50118
                   }
matLib['Sn119'] = {'M': 118.9033,
                   'A': 50,
                   'zaid': 50119
                   }
matLib['Sn120'] = {'M': 119.9021966,
                   'A': 50,
                   'zaid': 50120
                   }
matLib['Sn122'] = {'M': 121.90344,
                   'A': 50,
                   'zaid': 50122
                   }
matLib['Sn124'] = {'M': 123.90527,
                   'A': 50,
                   'zaid': 50124
                   }
matLib['Hf174'] = {'M': 173.94,
                   'A': 72,
                   'zaid': 72174
                   }
matLib['Hf176'] = {'M': 175.9414,
                   'A': 72,
                   'zaid': 72176
                   }
matLib['Hf177'] = {'M': 176.94322,
                   'A': 72,
                   'zaid': 72177
                   }
matLib['Hf178'] = {'M': 177.9436977,
                   'A': 72,
                   'zaid': 72178
                   }
matLib['Hf179'] = {'M': 178.9458151,
                   'A': 72,
                   'zaid': 72179
                   }
matLib['Hf180'] = {'M': 179.9465488,
                   'A': 72,
                   'zaid': 72180
                   }


def updateAtomicMass(matLib=matLib):
    ''' Update atomic masses and atomic numbers. '''
    for key in matLib.keys():
        zaidStr = str(matLib[key]['zaid'])
        try:
            atM = zaidStr[2:]
            if len(zaidStr) == 4:
                aN = zaidStr[0]
            else:
                aN = zaidStr[:2]
            matLib[key]['M'] = elements[int(aN)][int(atM)].mass
            matLib[key]['A'] = float(elements[int(aN)][int(atM)].number)
        except:
            print("Problem in updating properties for isotope " + str(key))
            if not matLib[key]['M'] or not matLib[key]['A']:
                print("Isotope: " + str(key) + "Supply isotope atomic mass and atomic number manually")
                sys.exit()
            else:
                print("Isotope: " + str(key) + ": Using default values for atomic mass and atomic number")
    return matLib
try:
    from periodictable import *
    matLib = updateAtomicMass(matLib)
except:
    print("WARNING: python package periodictable is not installed")
    print("Continuing with user defined atomic mass definitions")

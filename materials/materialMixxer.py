import sys
import numpy as np
from readXSdatabase import readXSFolder
from globalMatLib import matLib


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


def genMaterialDict(xsFolder='./hw2'):
    '''
    Updates the global matLib with all the xs data found in the XS directory
    '''
    myXsDict = readXSFolder(xsFolder)
    for matName in myXsDict.keys():
        props = myXsDict[matName]
        for prop in props:
            matLib[matName][prop] = myXsDict[matName][prop]
    print("Done Updating Materials.")


class mixedMat(object):
    '''
    Smeared material class.
    Computes macroscopic cross sections, diffusion coeffs, ect.
    Input number densities in #/b-cm
    Input cross sections in barns
    '''
    Na = 6.02214129e23  # avagadros constant

    def __init__(self, ndDict, wfFlag=False, matLib=matLib):
        # Initilize mixedMat dicts
        self.macroProp = {}
        self.nDdict = {}
        self.wf = {}
        self.af = {}
        self.afF = {}
        # Pull requested isotopes from material library
        try:
            materialData = {k: matLib[k] for k in ndDict.keys()}
        except:
            print("A Requested material was not found in material library. FATALITY.")
            sys.exit()
        self.microDat = materialData
        # If weight fracs are given, convert to atom frac and proceed as usual
        if wfFlag:
            self.wf = ndDict
            self.nD = self._wfToAf()
        else:
            self.nD = ndDict
        # Compute properties
        self._numberDensityTable()
        self._updateDB()

    def _updateDB(self):
        self.macroProp = {}
        self._computeAtomicFracs()
        self._computeMacroXsec()
        self._computeAvgProps()
        self._computeDensity()
        self._afToWf()

    def _numberDensityTable(self):
        '''
        Asscociate each isotope with number density.
        '''
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            self.nDdict[material] = self.nD[material]

    def _computeAtomicFracs(self):
        ''' Compute atomic fractions.
        Useful if material was specified by
        atom density or weight fractions '''
        fissionable_nD_sum = 0
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            self.af[material] = self.nDdict[material] / sum(self.nDdict.values())
            if 'chi' in data.keys():
                if sum(data['chi']) > 0:
                    fissionable_nD_sum += self.nDdict[material]
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            if 'chi' in data.keys():
                try:
                    self.afF[material] = self.nDdict[material] / fissionable_nD_sum
                except:
                    self.afF[material] = 0.0

    def _wfToAf(self):
        '''
        Converts weight fractions to atom fracs
        [g/gtot * mol/g * atoms/mol]  / atoms tot
        assuming gtot = 1
        '''
        atomSum = 0
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            atomSum += self.wf[material] * (1. / self.microDat[material]['M']) * self.Na
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            self.af[material] = self.wf[material] * (1. / self.microDat[material]['M']) * self.Na / atomSum
        return self.af

    def _afToWf(self):
        '''
        Converts atom fractions to weight fractions
        [atoms/atomstot * atomstot * mol/atom * g/mol] / g tot
        assuming atomstot = 1
        '''
        if hasattr(self, 'af'):
            wfSum = 0
            for i, (material, data) in enumerate(self.microDat.iteritems()):
                wfSum += self.af[material] * (1 / self.Na) * self.microDat[material]['M']
            for i, (material, data) in enumerate(self.microDat.iteritems()):
                self.wf[material] = self.af[material] * (1 / self.Na) * self.microDat[material]['M'] / wfSum
        else:
            self._computeAtomicFracs()
            self._afToWf()
        return self.wf

    def _computeMacroXsec(self):
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            for key, value in data.iteritems():
                if key in ['nufission', 'total', 'skernel']:
                    try:
                        self.macroProp['N' + key] += self.nDdict[material] * value
                    except:
                        self.macroProp['N' + key] = self.nDdict[material] * value
                else:
                    pass

    def _computeAvgProps(self):
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            for key, value in data.iteritems():
                if key in ['M']:
                    try:
                        self.macroProp[key] += self.af[material] * value
                    except:
                        self.macroProp[key] = self.af[material] * value
                elif key in ['nu', 'chi']:
                    try:
                        self.macroProp[key] += self.afF[material] * value
                    except:
                        self.macroProp[key] = self.afF[material] * value
                elif key in ['A']:
                    try:
                        self.macroProp[key] += self.af[material] * value
                        self.macroProp['mu_bar'] += self.af[material] * (2.0 / (3.0 * value))
                    except:
                        self.macroProp[key] = self.af[material] * value
                        self.macroProp['mu_bar'] = self.af[material] * (2.0 / (3.0 * value))
                else:
                    pass

    def _checkMaterialLib(self):
        '''
        Checks the incomming material libs for missing data.
        '''
        #requiredData = ['total', 'skernel', 'M', 'A']
        pass

    def _isFissile(self):
        ''' extra data for fissile materials '''
        # Just check elsewhere if a mixture has the 'chi' macroProp
        # if not, we know there is no fissile isotope in the mixture
        pass

    def _computeDensity(self):
        self.density = sum(self.macroProp['M'] * np.array(self.nDdict.values()) * 1.0e24 /
                           self.Na / np.array(self.af.values())) / len(self.af.values())

    def setDensity(self, density):
        ''' Supply material density in g/cc.  Updates atom densities accordingly. '''
        self.density = density  # g/cc
        # need keep atomic ratios the same, but update atom densities
        # such that the density specified is met
        for i, (material, data) in enumerate(self.microDat.iteritems()):
            # g/cc * #/mol * mol/g
            self.nDdict[material] = density * self.af[material] * self.Na / self.macroProp['M'] / 1.0e24
        self._updateDB()

    def __add__(self, other):
        '''
        Allows the mixing of predifined mixtures by mass:
        mix 0.2 U238 (by mass) with 0.8 lwtr (by mass) where the
        U238 and lwtr materials were constructed by mixing individual
        isotopes.
        '''
        # add number densities together
        for material, nDensity in other.nDdict.iteritems():
            try:
                self.nDdict[material] += nDensity
            except:
                # We've got a new isotope on our hands!
                self.nDdict[material] = nDensity
                self.microDat[material] = other.microDat[material]
        # update macro property database
        self._updateDB()
        return self

    def __mul__(self, const):
        '''
        Multiplication by weight fraction.
        '''
        if hasattr(self, 'density'):
            self.setDensity(self.density * const)
        else:
            print("Warning: must specify material density before weighting by mass fraction")
        return self

    def __rmul__(self, const):
        return self.__mul__(const)


if __name__ == "__main__":
    # Testing
    genMaterialDict()
    leuMat = mixedMat({'u235': 0.000818132,
                       'u238': 0.0251546})
    print(leuMat.density)
    modMat = mixedMat({'h1': 2 / 3.0,
                       'o16': 1 / 3.0})
    modMat.setDensity(1.0)
    print(modMat.nDdict)
    # Homogenize
    smear = 0.6 * modMat + 0.4 * leuMat
    print("homogenized props")
    print(smear.density)
    print(smear.nDdict)
    print("nufission")
    print(smear.macroProp['Nnufission'])
    print("chi")
    print(smear.macroProp['chi'])

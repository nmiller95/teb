import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table
from astropy.table import join
from astropy.units import UnitsWarning
import warnings
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import getopt
import requests
import xml.etree.ElementTree as ET

def getprofile(filtername, photon):
    fileroot = filtername.replace('/','.')
    # First look for user-defined transmission function in FITS file
    filterfile = os.path.join('cache','fps',fileroot+'.fits')
    if os.path.exists(filterfile):
        filtertable = Table.read(filterfile)
        if photon == -1:
            return filtertable
        photon_ = filtertable.meta['PHOTON']
        if (photon == None):
            return filtertable, photon_
        if photon and not photon_:
            print(' * Warning - Detector type Photon counter set by user *'
                    ' inconsistent with 0:Energy counter value from meta data')
            return filtertable, photon
        if not photon and photon_:
            print(' * Warning - Detector type Energy counter set by user *'
                    ' inconsistent with 1:Photon counter value from meta data')
            return filtertable, photon
        return filtertable, photon
    else:
        filterfile = os.path.join('cache','fps',fileroot+'.xml')
        try:
            filtertable = Table.read(filterfile)
            root = ET.parse(filterfile).getroot()
        except FileNotFoundError:
            cachedir = os.path.join('cache','fps')
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)
            url = 'https://svo2.cab.inta-csic.es/theory/fps/fps.php'
            query_parameters = {'ID':filtername}
            response = requests.get(url, params=query_parameters)
            root = ET.fromstring(response.content)
            if root.find('INFO').attrib['value'] != 'OK':
                msg = root.find('INFO').find('DESCRIPTION').text
                raise AttributeError(msg)
            with open(filterfile, mode='wb') as fp:
                fp.write(response.content)
            filtertable =  Table.read(filterfile)
        if photon == -1:
            return filtertable
    
        DetectorType = None
        for par in root.find('RESOURCE').find('TABLE').findall('PARAM'):
            if par.attrib['name'] ==  'DetectorType':
                DetectorType = int(par.attrib['value'])
        if (DetectorType is None) and (photon is None):
            raise LookupError('DetectorType not found in VOTable')
        elif DetectorType == 0:
            if (photon == None) or not photon:
                #print(f' Detector type. 0:Energy counter ({DetectorType})')
                photon = False
            else:
                print(' * Warning - Detector type Photon counter set by user *'
                    ' inconsistent with 0:Energy counter value from meta data')
        elif DetectorType == 1:
            if (photon == None) or photon:
                #print(f' Detector type. 1:Photon counter ({DetectorType})')
                photon = True
            else:
                print(' * Warning - Detector type Energy counter set by user *'
                    ' inconsistent with 0:Photon counter value from meta data')
        else:
            raise AttributeError(f'Unknown DetectorType ({DetectorType})')
    return filtertable, photon

def getspec(fitsfile):
    savefile = os.path.join('cache','CALSPEC',fitsfile)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UnitsWarning)
            spec = Table.read(savefile)
    except FileNotFoundError:
        url = 'https://archive.stsci.edu/'
        url += 'hlsps/reference-atlases/cdbs/current_calspec/'+fitsfile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UnitsWarning)
            spec = Table.read(url)
            spec.write(savefile)
    return spec

def _xmean_func(sigma_x, y, yerr, prior):
    w = 1/(yerr**2 + sigma_x**2)
    ybar = np.sum(y*w)/np.sum(w)
    if prior is None:
        return np.sum(w*(y-ybar)**2 - np.log(w))
    else:
        return (np.sum(w*(y-ybar)**2 - np.log(w)) +
                ((sigma_x-prior[0])/prior[1])**2  )

def xmean(y,yerr,prior=None):
    # Gaussian prior N(prior[0],prior[1])
    w = 1/yerr**2
    ybar = np.sum(y*w)/np.sum(w)
    chisq_r = np.sum(w*(y-ybar)**2)/(len(y)-1)
    if chisq_r > 1:
       r = minimize_scalar(_xmean_func, [0,np.std(y)],
                           args=(y, yerr, prior))
       sigma_x = abs(r.x)
       w = 1/(yerr**2 + sigma_x**2)
       ybar = np.sum(y*w)/np.sum(w)
    else:
        sigma_x = 0
    sigma_ybar = np.sqrt(1/np.sum(w))
    return ybar, sigma_ybar, sigma_x

def inputs(argv):
    def usage(short=True):
        print('Usage: calspec.py [-v] [-e] [-p] [-f] [-n] filter')
        print(' ')
        print(' The photometric system and filter to be calibrated must be ')
        print(' specified using a name from the Spanish Virtual Observatory')
        print(' (SVO) Filter profile service e.g. "GAIA/GAIA3.G" or an xml')
        print(' file in cache/fps/containing filter profile information')
        print(' in the same format as the SVO Filter profile service.')
        print(' ')
        print(' The stars to be used for calibration must be listed in the  ')
        print(' hfile calspec/<filterfile>.csv, where <filterfile> is the same')
        print(' as the SVO filtername but with "/" replace by ".", e.g.     ')
        print(' "calspec/GAIA.GAIA3.G.csv"                                   ')
        print(' ')
        print(' The CSV file must include three columns with the following  ')
        print(' names and information:')
        print('  - name: Star name as listed in column 5 of Table 1a on the ')
        print('          CALSPEC web page, e.g. "109vir".')
        print('  - mag: observed magnitude in the filter to be calibrated')
        print('  - e_mag: standard error on the observed magnitude.')
        print(' ')
        print(' Use the option "-v" if the observed magnitudes are Vega')
        print(' magnitudes, otherwise they are assumed to be AB magnitudes.')
        print(' ')
        print(' By default, the detector type (photon counter or energy ')
        print(' counter) is taken from the SVO XML file meta data. To over-')
        print(' ride this behaviour, use the option "-e" to assume that the')
        print(' detector used measures energy or use the option "-p" to assume')
        print(' that the detector counts photons. The option "-e" should also')
        print(' be used if the transmission data from SVO includes the ')
        print(' correction needed to account for photon-counting detectors ')
        print(' when energy integration is used to calculate the mean flux.')
        print(' (See, e.g. Casagrande & VandenBerg, MNRAS 444, 392, 2014)')
        print(' ')
        print(' Use the "-o" option to specify an output CSV file that ')
        print(' may be useful to find trends with magnitude, colour, etc. ')
        print(' in the residuals. ')
        print(' ')
        print(' Use the option "-f" to specify a file containing data for a')
        print(' new filter. The file must be a simple two-column ASCII file')
        print(' with wavelength in Angstrom in the first column and system')
        print(' response (tranmission x sensitivity) in the second column. ')
        print(' Either the option "-p" or the option "-e" must be specified.')
        print(' The new filter data will be saved in the file')
        print(' "cache/fps/<filter>.fits", where <filter> is the new filter')
        print(' name specified on the command line.')
        print(' Use the "-n" option to skip the zero-point calculation if the')
        print(' filter will be used to calculate flux ratios only, e.g.')
        print(' $ python calspec.py -n -p -f calspec/u350.dat User/IUE.u350')
        print(' ')
        print(' The option "-u" will calculate a set of zero-points for')
        print(' the Stromgren b-y, m1 and c1 indices based on the observed')
        print(' uvby photometry of stars in the file calspec/uvby.csv.')
        print(' This CSV file must have the following column: ')
        print('  - name: Star name as listed in column 5 of Table 1a on the ')
        print('          CALSPEC web page, e.g. "109vir".')
        print('  - by: observed b-y colour ')
        print('  - e_by: standard error on the observed b-y colour.')
        print('  - m1: observed m1 index ')
        print('  - e_m1: standard error on the observed m1 index.')
        print('  - c1: observed c1 index ')
        print('  - e_c1: standard error on the observed c1 index.')
        print(' No other options apply if the "-u" option is used.')
        print(' ')
        print(' The final line(s) of output are the data needed to populate')
        print(' the database of photometric systems config/database.csv with')
        print(' columns:')
        print(' filtername,zp,zp_err,photon,vega,magmin,magmax,pivot,sigma_x ')
        print(' ')
    vega = False
    uvby = False
    photon = None
    output = None
    file = None
    nozp = False
    try:
        opts, args = getopt.getopt(argv, "hpevuno:f:", [
            "help","photon-counter","energy-counter","vega","uvby",
            "no-zero-point", "output","file"])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)


    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        if opt in ("-o", "--output"):
            output = arg
        if opt in ("-u","--uvby"):
            uvby = True
    if uvby:
        return None, None, None, output, uvby

    if len(args) != 1:
        print('Usage: calspec.py [-h] [-p] [-e] [-v] [-u] [-n] [-o output]'
              ' [-f file] filter')
        sys.exit(1)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        if opt in ('-p','--photon-counter'):
            if photon is not None:
                raise ValueError('Cannot specify both --energy-counter [-e]'
                                 ' and --photon-counter [-p]')
            photon = True
        if opt in ('-e','--energy-counter'):
            if photon is not None:
                raise ValueError('Cannot specify both --energy-counter [-e]'
                                 ' and --photon-counter [-p]')
            photon = False
        if opt in ("-v","--vega"):
            vega = True
        if opt in ("-o", "--output"):
            output = arg
        if opt in ("-f", "--file"):
            file = arg
        if opt in ("-n", "--no-zero-point"):
            nozp = True
    if nozp and file is None:
        m="--no-zero-point [-n] not applicable without option --file [-f]"
        raise ValueError(m)
    if file is not None and (photon is None):
        raise ValueError('Specify either --energy-counter [-e] or'
                                 ' --photon-counter [-p]')

    return args[0], vega, photon, output, uvby, file, nozp

#-------------------------

if __name__ == "__main__":
    print("""
    calspec -- a python tool to calculate photometric zero points

    Written by P. F. L. Maxted (p.maxted@keele.ac.uk)
    """)

    filtername, vega, photon, output, uvby, file, nozp = inputs(sys.argv[1:])

    if file is not None:
        filtertable = Table.read(file,format='ascii',
                            names=['Wavelength','Transmission'])
        print(f'\n New filter data loaded from {file}')
        filtertable.meta['PHOTON'] = photon
        fileroot = filtername.replace('/','.')
        filterfile = os.path.join('cache','fps',fileroot+'.fits')
        filtertable.write(filterfile, overwrite=True)

        if nozp:
            print(f' New filter data written to {filterfile}')
            wave = filtertable['Wavelength']
            resp = filtertable['Transmission']
            wmin = min(wave)
            wmax = max(wave)
            print(f' Wavelength range = {wmin:0.1f} - {wmax:0.1f} A')
            pivot = (np.sqrt(simpson(resp*wave, x=wave) /
                             simpson(resp/wave, x=wave) ))
            print(f' Pivot wavelength = {pivot:0.1f} A')
            print('')
            sys.exit()

    # CALSPEC star data and model names
    Table1a = Table.read('calspec/Table1a.csv')
    Table1a['Model'] = Table1a['Model'].filled('')

    if uvby:
        starfile = os.path.join('calspec','uvby.csv')
        starlist = Table.read(starfile)
        T = {}
        wmin = 1e20
        wmax = 0
        for b in ['u','v','b','y']:
            # photon irrelevant for these narrow bands, but assume False
            # Stromgren system is Vega magnitudes
            filtertable = getprofile(f'Generic/Stromgren.{b}', -1)
            wave = filtertable['Wavelength']
            resp = filtertable['Transmission']
            resp /= simpson(resp, x=wave) # Normalisation
            wmin = min([wmin, min(wave)])
            wmax = max([wmax, max(wave)])
            T[b] = interp1d(wave, resp, bounds_error=False, fill_value=0)
        by = []
        m1 = []
        c1 = []
        for row in starlist:
            name = str(row['name'])
            row1a = Table1a[Table1a['Name'] == name]
            if len(row1a) == 0:
                m = f'Star name {name} not found in Table 1a.'
                raise AttributeError(m)
            fitsfile = name + row1a['STIS'][0].replace('*','')+'.fits'

            spec = getspec(fitsfile)
            wave = spec['WAVELENGTH']   # A
            flux = spec['FLUX']  # erg s-1 cm-2 A-1.
            qual = spec['DATAQUAL']  # erg s-1 cm-2 A-1.
            ok = (flux > 0) & (qual == 1)
            wave = wave[ok]
            flux = flux[ok]
            if (wave.min() > wmin) | (wave.max() < wmax):
                print(f'* Spectrum {fitsfile} does cover uvby wavelength range'
                       ' - skipped *')
                by.append(np.nan)
                m1.append(np.nan)
                c1.append(np.nan)
                continue
            u = -2.5*np.log10(simpson(flux*T['u'](wave), x=wave))
            v = -2.5*np.log10(simpson(flux*T['v'](wave), x=wave))
            b = -2.5*np.log10(simpson(flux*T['b'](wave), x=wave))
            y = -2.5*np.log10(simpson(flux*T['y'](wave), x=wave))
            by.append(row['by'] - (b-y))
            m1.append(row['m1'] - ((v-b)-(b-y)))
            c1.append(row['c1'] - ((u-v)-(v-b)))

        i = np.isfinite(by)
        n = sum(i)
        if sum(i) < 3:
            m='Too few measurements for zero point calculation'
            raise ValueError(m)
        y = np.array(by)[i]
        yerr = starlist['e_by'][i]
        zp_by,e_by,sigma_by = xmean(y,yerr)
        e_by = max([e_by,0.0001])
        chisq_by = np.sum((y-zp_by)**2/yerr**2)
        y = np.array(m1)[i]
        yerr = starlist['e_m1'][i]
        zp_m1,e_m1,sigma_m1 = xmean(y,yerr)
        e_m1 = max([e_m1,0.0001])
        chisq_m1 = np.sum((y-zp_m1)**2/yerr**2)
        y = np.array(c1)[i]
        yerr = starlist['e_c1'][i]
        zp_c1,e_c1,sigma_c1 = xmean(y,yerr)
        e_c1 = max([e_c1,0.0001])
        chisq_c1 = np.sum((y-zp_c1)**2/yerr**2)
        print(' ')
        print(' Name               b-y        o-c        m1        o-c        '
              'c1        o-c ')
        print(' ----------------------------------------------------------'
              '--------------------')
        for i,r in enumerate(starlist):
            if np.isfinite(by[i]):
                l = f' {r["name"]:<14} '
                l += f' {r["by"]:6.3f}+-{r["e_by"]:5.3f} {by[i]-zp_by:6.3f}'
                l += f' {r["m1"]:6.3f}+-{r["e_m1"]:5.3f} {m1[i]-zp_m1:6.3f}'
                l += f' {r["c1"]:6.3f}+-{r["e_c1"]:5.3f} {c1[i]-zp_c1:6.3f}'
                print(l)
        print(' ----------------------------------------------------------'
              '--------------------')

        print(' ')
        print(f' b-y zero point:  {zp_by:0.4f} +/- {e_by:0.4f}')
        print(f' Estimated excess variance = ({sigma_by:0.4f})^2')
        i = np.isfinite(by)
        bymin = np.floor(100*np.nanmin(starlist['by'][i]))/100
        bymax = np.ceil(100*np.nanmax(starlist['by'][i]))/100
        print(f' Valid b-y range = {bymin:0.2f} - {bymax:0.2f}')
        print(f' chi-squared = {chisq_by:0.2f}')
        print(f' N = {sum(i)}')
        print(' ')
        print(f' m1 zero point:  {zp_m1:0.4f} +/- {e_m1:0.4f}')
        print(f' Estimated excess variance = ({sigma_m1:0.4f})^2')
        i = np.isfinite(m1)
        m1min = np.floor(100*np.nanmin(starlist['m1'][i]))/100
        m1max = np.ceil(100*np.nanmax(starlist['m1'][i]))/100
        print(f' Valid m1 range = {m1min:0.2f} - {m1max:0.2f}')
        print(f' chi-squared = {chisq_m1:0.2f}')
        print(f' N = {sum(i)}')
        print(' ')
        print(f' c1 zero point:  {zp_c1:0.4f} +/- {e_c1:0.4f}')
        print(f' Estimated excess variance = ({sigma_c1:0.4f})^2')
        i = np.isfinite(c1)
        c1min = np.floor(100*np.nanmin(starlist['c1'][i]))/100
        c1max = np.ceil(100*np.nanmax(starlist['c1'][i]))/100
        print(f' Valid c1 range = {c1min:0.2f} - {c1max:0.2f}')
        print(f' chi-squared = {chisq_c1:0.2f}')
        print(f' N = {sum(i)}')
        print(' ')
        print(' Data for config/database.csv:')
        print(' ')
        l = f'by,{zp_by:0.4f},{e_by:0.4f},False,True'
        l += f',{bymin:0.2f},{bymax:0.2f},,{sigma_by:0.4f}'
        print(l)
        l = f'm1,{zp_m1:0.4f},{e_m1:0.4f},False,True'
        l += f',{m1min:0.2f},{m1max:0.2f},,{sigma_m1:0.4f}'
        print(l)
        l = f'c1,{zp_c1:0.4f},{e_c1:0.4f},False,True'
        l += f',{c1min:0.2f},{c1max:0.2f},,{sigma_c1:0.4f}'
        print(l)
        exit(0)
        # End of uvby calibration loop

    fileroot = filtername.replace('/','.')
    starfile = os.path.join('calspec',fileroot+'.csv')
    starlist = Table.read(starfile)
    iz = starlist['e_mag'] <= 0
    if sum(iz) > 0:
        raise ValueError('Input star data file contains invalid e_mag values')

    print(' Filter: ',filtername)
    filtertable,photon = getprofile(filtername, photon)

    wave = filtertable['Wavelength']
    resp = filtertable['Transmission']
    wmin = min(wave)
    wmax = max(wave)
    print(f' Wavelength range = {wmin:0.1f} - {wmax:0.1f} A')
    if vega:
        print(' Observed magnitudes are Vega magnitudes.')
    else:
        print(' Observed magnitudes are AB magnitudes.')
    if photon:
        print(' Assuming photon-counting detector.')
    else:
        print(' Assuming energy-measuring detector.')
    pivot = np.sqrt(simpson(resp*wave, x=wave) / simpson(resp/wave, x=wave))
    print(f' Pivot wavelength = {pivot:0.1f} A')

    # Normalize spectral response function here
    if photon: 
        resp /= simpson(wave*resp, x=wave)
    else:      
        resp /= simpson(resp, x=wave)
    T = interp1d(wave, resp, bounds_error=False, fill_value=0)

    zps = []
    patched = []
    c_ = 2.99792e8
    for row in starlist:
        name = str(row['name'])
        row1a = Table1a[Table1a['Name'] == name]
        if len(row1a) == 0:
            m = f'Star name {name} not found in Table 1a.'
            raise AttributeError(m)
        fitsfile = name + row1a['STIS'][0].replace('*','')+'.fits'

        spec = getspec(fitsfile)
        wave = spec['WAVELENGTH']   # A
        flux = spec['FLUX']  # erg s-1 cm-2 A-1.
        qual = spec['DATAQUAL']  # erg s-1 cm-2 A-1.
        ok = (flux > 0) & (qual == 1)
        wave = wave[ok]
        flux = flux[ok]
        if (wave.min() > wmin) | (wave.max() < wmax):
            i = (wave <= wmax) & (wave >= wmin)
            if sum(i) > 0:
                coverage = np.ptp(wave[i])/(wmax-wmin)
            else:
                coverage = 0
            if coverage < 0.8:
                print(f'* Spectrum {fitsfile} does covers < 80% of filter'
                       'bandpass - skipped *')
                zps.append(np.nan)
                patched.append(-1)
                continue

            if row1a['Model'][0] == '':
                print(f' * No model available to estimate UV flux for {name}'
                       ' - skipped *')
                zps.append(np.nan)
                patched.append(-1)
                continue

            modelfile = name + row1a['Model'][0].replace('*','')+'.fits'
            model = getspec(modelfile)
            wmodel = model['WAVELENGTH']
            fmodel = model['FLUX']
            pflag = 0
            if wave.min() > wmin:
                if (wmodel.min() > wmin):
                    print(f' * Model {modelfile} does not cover filter '
                        'wavelength range - skipped *')
                    zps.append(np.nan)
                    patched.append(-1)
                    continue
                i = (wmodel < wave.min()) 
                wmodel = wmodel[i]
                fmodel = fmodel[i]
                wave = np.hstack([wmodel, wave])
                flux = np.hstack([fmodel, flux])
                pflag += 1
            if wave.max() < wmax:
                if (wmodel.max() < wmin):
                    print(f' * Model {modelfile} does not cover filter '
                           'wavelength range - skipped *')
                    zps.append(np.nan)
                    patched.append(-1)
                    continue
                i = (wmodel > wave.max()) 
                wmodel = wmodel[i]
                fmodel = fmodel[i]
                wave = np.hstack([wave, wmodel])
                flux = np.hstack([flux, fmodel])
                pflag += 2
            patched.append(pflag)
        else:
            patched.append(0)
        if (wave.min() > wmin) or (wave.max() < wmax):
            print(f'Spectrum {fitsfile} does not cover filter wavelength '
                   'range - skipped *')
            zps.append(np.nan)
            continue
        
        obs_mag = row['mag']
        if vega:
            if photon:
                f_lambda = simpson(wave*flux*T(wave), x=wave)
            else:
                f_lambda = simpson(flux*T(wave), x=wave)
            if (f_lambda <= 0):
                print(f' * Negative / zero flux for star {name} - skipped *')
                zps.append(np.nan)
                continue
            zps.append(obs_mag+2.5*np.log10(f_lambda))
        else:
            # Using Bessel & Murphy, 2012 PASP 124 140, equation (A15)
            if photon:
                f_nu = simpson(wave*flux*T(wave), x=wave)*1e-10*(pivot)**2/c_
            else:
                f_nu = simpson(flux*T(wave), x=wave)*1e-10*pivot**2/c_
            if (f_nu <= 0):
                print(f' * Negative / zero flux for star {name} - skipped *')
                zps.append(np.nan)
                continue
            zps.append(obs_mag+2.5*np.log10(f_nu))

    i = np.isfinite(zps)
    n = sum(i)
    if sum(i) < 3:
        raise ValueError('Too few measurements for zero point calculation')
    y = np.array(zps)[i]
    yerr = starlist['e_mag'][i]
    zp,zp_err,sigma_x = xmean(y,yerr)
    zp_err = max([zp_err,0.0001])
    chisq = np.sum((y-zp)**2/yerr**2)


    print(' ')
    print(' Name                mag_obs       mag_syn (O-C)  Notes')
    print(' ------------------------------------------------------')
    for i,row in enumerate(starlist):
        if np.isfinite(zps[i]):
            mag = row['mag']
            e_mag = row['e_mag']
            syn = mag-zps[i] + zp
            name = row['name']
            res = mag - syn
            l= f' {name:<16} {mag:7.3f} +- {e_mag:5.3f}  {syn:7.3f} {res:+6.3f}'
            if patched[i] > 0:
                l += f'  {patched[i]}'
            print(l)
    print(' -------------------------------------------------------')
    if 1 in patched:
        print(' 1 UV flux extended with model')
    if 2 in patched:
        print(' 2 IR flux extended with model')
    if 3 in patched:
        print(' 3 UV and IR flux extended with model')

    print(' ')
    print(f' Zero point:  {zp:0.4f} +/- {zp_err:0.4f}')
    print(f' Estimated excess variance = ({sigma_x:0.3f} mag)^2')
    i = np.isfinite(zps)
    magmin = np.floor(10*np.nanmin(starlist['mag'][i]))/10
    magmax = np.ceil(10*np.nanmax(starlist['mag'][i]))/10
    print(f' Valid magnitude range = {magmin:0.2f} - {magmax:0.1f}')
    print(f' chi-squared = {chisq}')
    print(f' N = {n}')
    print(' ')
    print(' Data for config/database.csv:')
    print(' ')
    l = f'{filtername},{zp:0.4f},{zp_err:0.4f},{photon},{vega}'
    l += f',{magmin:0.1f},{magmax:0.1f},{pivot:0.1f},{sigma_x:0.4f}'
    print(l)

    if output:
        t = join(Table1a,starlist,keys_left='Name',keys_right=['name'])
        t['mag_syn'] = starlist['mag'] - np.array(zps) + zp
        t['resid'] = starlist['mag'] - t['mag_syn']
        t['uvflag'] = patched
        t.write(output,format='csv',overwrite=True)
        print( 'Detailed results written to',output)



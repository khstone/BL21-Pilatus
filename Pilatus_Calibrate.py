import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from Pilatus_Calibration_setup import *

def read_TIFF(file):
    print "Reading TIFF file here..."
    try:
        im = open(file, 'rb')
        im.seek(4096)	# skip the first 4096 bytes of header info for TIFF images
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)  #for the way mounted at old BL2-1
        print np.shape(arr)
        print len(arr)
        return arr
    except:
        print "Error reading file: %s" % file
        return None

def read_RAW(file):
    print "Reading RAW file here..."
    try:
        im = open(file, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)  #for the way mounted at old BL2-1
        return arr
    except:
        print "Error reading file: %s" % file
        return None

def csvread(filename):  
    print "Reading CSV file here..."
    csv = open(filename)
    line = csv.readline()
    temp = line.split(',')
    xi = 1
    yi = 4
    i0i = 3
    x = []
    y = []
    i0 = []
    line = csv.readline()
    while line:
        temp = line.split(",")
        x = np.append(x, float(temp[xi]))
        y = np.append(y, float(temp[yi]))       
        i0 = np.append(i0, float(temp[i0i]))
        line = csv.readline()
    csv.close()
    return x, y, i0

def gauss_linbkg(x, m, b, x0, intint, fwhm):
    return m*x + b + intint*(2./fwhm)*np.sqrt(np.log(2.)/np.pi)*np.exp(-4.*np.log(2.)*((x-x0)/fwhm)**2)
    
def Gauss_fit(x, y):
    pguess = [0, 0, np.argmax(y), np.max(y), 5.0]  # linear background (2), pos, intensity, fwhm
    popt, pcov = curve_fit(gauss_linbkg, x, y, p0=pguess)
    return popt
    
def simple_line(x, m, b):
    return m*x + b

# Read CSV file, get step size and number of points in scan
x, y, i0 = csvread(csv_path + csv_name)
num_points = len(x)
calib_tth_steps = abs(x[1] - x[0])
x = []
y = []
i0 = []

# Read images, take line cut, fit feature position for calibration scan
pks = []
for i in xrange(0, num_points):
    filename = data_path + calib_name + str(i).zfill(4) + ".raw"
    data = read_RAW(filename)
    x = np.arange(0, np.shape(data)[1])
    y = data[db_pixel[1], :]
    y += data[db_pixel[1] + 1, :]
    y += data[db_pixel[1] - 1, :]
    y += data[db_pixel[1] + 2, :]
    y += data[db_pixel[1] - 2, :]
    popt = Gauss_fit(x, y)
    pks = np.append(pks, popt[2])

# Fit line to the extracted peak positions and determine the sample to detector distance
x = np.arange(num_points)
lin_fit, pcov = curve_fit(simple_line, pks, x*calib_tth_steps + 0.00)
det_R = 1.0/np.tan(abs(lin_fit[0])*np.pi/180.0)     # sample to detector distance in pixels
print "Sample to detector distance in pixels = " + str(det_R)
print "Sample to detector distance in mm = " + str((det_R * pix_size / 1000.0))
plt.figure()
plt.plot(pks, x*calib_tth_steps, 'b.')
plt.plot(pks, lin_fit[0]*pks + lin_fit[1], 'r-')

outname = csv_path + csv_name[:-4] + "_calib.cal"
outfile = open(outname, "w")
outfile.write("direct_beam_x \t %i\n"  % db_pixel[0])
outfile.write("direct_beam_y \t %i\n" % db_pixel[1])
outfile.write("Sample_Detector_distance_pixels \t %15.6G\n" % det_R)
outfile.write("Sample_Detector_distance_mm \t %15.6G" % (det_R * pix_size / 1000.0))
outfile.close()

plt.show()

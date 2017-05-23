import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
from matplotlib import pyplot as plt
import gc

from Pilatus_Integration_setup import *
mult = 10000000.0

def read_TIFF(file):
    print "Reading TIFF file here..."
    try:
        im = open(file, 'rb')
        im.seek(4096)	# skip the first 4096 bytes of header info for TIFF images
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        arr = np.fliplr(arr)  #for the way mounted at BL2-1
        return arr
    except:
        print "Error reading file: %s" % file
        return None

def read_RAW(file, mask = True):
    print "Reading RAW file here..."
    try:
        im = open(file, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        arr = np.fliplr(arr)               # for the way mounted at BL2-1
        if mask:
            for i in xrange(0, 10):
                arr[:,i] = -2.0
            for i in xrange(477, 487):
                arr[:,i] = -2.0
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

def SPECread(filename, scan_number):
    print "Reading SPEC file here..."
    tth = []
    i0 = []
    spec = open(filename)
    for line in spec:
        if "#O" in line and "TwoTheta" in line:  #find which line has the 2theta position
            temp = line.split()
            tth_line = temp[0][2]
            for i in range(0, len(temp)):
                if temp[i] == "TwoTheta":	#find where in that line the 2theta position is listed
                    tth_pos = i
                    break
            break
    for line in spec:
        if "#S" in line:
            temp = line.split()
            if int(temp[1]) == scan_number:
                break
    for line in spec:
        if "#P" + str(tth_line) in line:
            temp = line.split()
            tth_start = float(temp[tth_pos])
            break
    for line in spec:
        if "#L" in line:
            motors = line.split()[1:]
            if "TwoTheta" not in line:
                tth_motor_bool = False
                print "2theta is not scanned..."
            else:
                tth_motor_bool = True
                tth_motor = motors.index("TwoTheta")
            i0_motor = motors.index("Monitor")
            break
    for line in spec:
        try:
            temp = line.split()
            if tth_motor_bool:
                tth = np.append(tth, float(temp[tth_motor]))
            else:
                tth = np.append(tth, tth_start)
            i0 = np.append(i0, float(temp[i0_motor]))
        except:
            break
    spec.close()
    return tth, i0

####################################################################################
#  Read calibration file and get calibration constants
####################################################################################
cal = open(calib_path + calib_name)
line = cal.readline()
db_x = int(line.split()[-1])
line = cal.readline()
db_y = int(line.split()[-1])
line = cal.readline()
det_R = float(line.split()[-1])
cal.close()
db_pixel = [487-db_x, db_y]  #This line is important for the way the detector is mounted at BL2-1

# Map each pixel into cartesian coordinates (x,y,z) in number of pixels from sample for direct beam conditions (2-theta = 0)
# We only need to do this once, so we can be inefficient about it
filename = image_path + user + spec_name + "_scan" + str(scan_number) + "_" + str(0).zfill(4) + ".raw"
data = read_RAW(filename)

xyz_map = np.empty([np.shape(data)[0], np.shape(data)[1], 3])
for i in xrange(0, np.shape(data)[0]):
    print i
    for j in xrange(0, np.shape(data)[1]):
        xyz_map[i,j] = [i-db_pixel[1], j-db_pixel[0], det_R]
print np.shape(xyz_map)

def rotate_operation(map, tth):
    # Apply a rotation operator to map this into cartesian coordinates (x',y',z') when 2-theta != 0
    # We should be efficient about how this is implemented
    xyz_map_prime = np.empty_like(map)
    tth *= np.pi/180.0
    rot_op = [[1.0, 0.0, 0.0], [0.0, np.cos(tth), np.sin(tth)], [0.0, -1.0*np.sin(tth), np.cos(tth)]]
    print np.shape(data)
    for i in xrange(0, np.shape(data)[0]):
        for j in xrange(0, np.shape(data)[1]):
            xyz_map_prime[i, j] = np.dot(rot_op, map[i, j])
    return xyz_map_prime
        
def cart2sphere(map):
    # Convert the rotated cartesian coordinate map to spherical coordinates
    # This should also be efficiently implemented
    tth_map = np.empty_like(data, dtype=float)
    for i in xrange(0, np.shape(data)[0]):
        for j in xrange(0, np.shape(data)[1]):
            tth_map[i, j] = np.arctan(np.sqrt(map[i, j, 0]**2 + map[i, j, 1]**2)/map[i, j, 2])*180.0/np.pi
    return tth_map%180.0

#######################################################################
###      Start scan information and integration here...
#######################################################################
tth, i0 = SPECread(spec_path + spec_name, scan_number)
x = []
y = []
xmax_global = 0.0  #set this to some small number so that it gets reset with the first image
xmin_global = 180.0
stepsize = 0.01
bins = np.arange(0.0, 180.0, stepsize)
digit_y = np.zeros_like(bins)
digit_norm = np.zeros_like(bins)
for k in range(0, len(tth)):
    x = []
    y = []
    print k
    print tth[k]
    filename = image_path + user + spec_name + "_scan" + str(scan_number) + "_" + str(k).zfill(4) + ".raw"
    print filename
    data = read_RAW(filename)
    xyz_map_prime = rotate_operation(xyz_map, tth[k])
    tth_map = cart2sphere(xyz_map_prime)
    for j in xrange(0, np.shape(data)[0]):
        x = np.append(x, tth_map[j])
        y = np.append(y, data[j]/i0[k])
    xmax = np.max(x)
    xmin = np.min(x)
    xmax_global = np.max([xmax, xmax_global])
    xmin_global = np.min([xmin, xmin_global])
    sort_index = np.argsort(x)
    inds = np.digitize(x, bins)
    for i in xrange(0, len(inds)):
        if y[i] < 0.0:
            continue
        digit_norm[inds[i]] += 1
        digit_y[inds[i]] += y[i]
        
nonzeros = np.nonzero(digit_norm)
interp = interpolate.InterpolatedUnivariateSpline(bins[nonzeros], digit_y[nonzeros]/digit_norm[nonzeros])

interpbins = np.arange(min(bins[nonzeros]), max(bins[nonzeros]), stepsize)
interpy = interp(interpbins)

outname = spec_path + spec_name + "scan" + str(scan_number) + ".xye"
outfile = open(outname, "w")
for i in xrange(0, len(interpbins)):
    outfile.write(str(interpbins[i]) + "\t" + str(mult * interpy[i]) + "\t" + str(np.sqrt(mult * interpy[i])) + "\n")
outfile.close()

plt.figure()
plt.plot(interpbins, interpy)

plt.figure()
plt.plot(bins, digit_norm)

plt.show()

import numpy as np
import scipy.interpolate as interpolate
import matplotlib
import matplotlib.pyplot as plt
import math

stepsize = 0.005   #define this here so that it may be reset by Pilatus_Integration_setup if desired
from Pilatus_Integration_setup import *
mult = 10000000.0

def read_TIFF(file):
    print("Reading TIFF file here...")
    try:
        im = open(file, 'rb')
        im.seek(4096)	# skip the first 4096 bytes of header info for TIFF images
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)  #uncomment if detector is mounted upside down
        return arr
    except:
        print("Error reading file: %s" % file)
        return None

def read_RAW(file, mask = True):
    print("Reading RAW file here...")
    try:
        im = open(file, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)               #uncomment if detector is mounted upside down
        if mask:
            for i in range(0, 10):
                arr[:,i] = -2.0
            for i in range(477, 487):
                arr[:,i] = -2.0
        return arr
    except:
        print("Error reading file: %s" % file)
        return None

def SPECread(filename, scan_number):
    print("Reading SPEC file here...")
    tthstring = "TwoTheta"   #define motor name in SPEC file
    tth = []
    i0 = []
    spec = open(filename)
    for line in spec:
        if "#O" in line and tthstring in line:  #find which line has the 2theta position
            temp = line.split()
            tth_line = temp[0][2]
            for i in range(0, len(temp)):
                if temp[i] == tthstring:	#find where in that line the 2theta position is listed
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
            if tthstring not in line:
                tth_motor_bool = False
                print("2theta is not scanned...")
            else:
                tth_motor_bool = True
                tth_motor = motors.index(tthstring)
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
db_pixel = [db_x, db_y]
#db_pixel = [487-db_x, db_y]  #This line is important for the way the detector is mounted at BL2-1

# Map each pixel into cartesian coordinates (x,y,z) in number of pixels from sample for direct beam conditions (2-theta = 0)
# We only need to do this once, so we can be inefficient about it
data = np.zeros((195, 487))

tup = np.unravel_index(np.arange(len(data.flatten())), data.shape)
xyz_map = np.vstack((tup[0], tup[1], det_R*np.ones(tup[0].shape)))
xyz_map -= [[db_pixel[1]], 
            [db_pixel[0]], 
            [0]]

def rotate_operation(map, tth):
    # Apply a rotation operator to map this into cartesian coordinates (x',y',z') when 2-theta != 0
    # We should be efficient about how this is implemented
    xyz_map_prime = np.empty_like(map)
    tth *= np.pi/180.0
    rot_op = np.array([[1.0, 0.0, 0.0], 
                       [0.0, np.cos(tth), np.sin(tth)], 
                       [0.0, -1.0*np.sin(tth), np.cos(tth)]])
    xyz_map_prime = np.matmul(rot_op, map)
    return xyz_map_prime
        
def cart2sphere(map):
    # Convert the rotated cartesian coordinate map to spherical coordinates
    # This should also be efficiently implemented
    tth_map = np.empty_like(data, dtype=float).flatten()
    _r = np.sqrt((map[:2,:]**2).sum(axis=0))
    tth_map = np.arctan(_r/map[2, :])*180.0/np.pi
    tth_map = tth_map.reshape(data.shape)
    return tth_map%180.0

#######################################################################
###      Start scan information and integration here...
#######################################################################
if np.isscalar(scan_number):
    scan_number = [scan_number]
for scan_num in scan_number:
    tth, i0 = SPECread(spec_path + spec_name, scan_num)
    x = []
    y = []
    xmax_global = 0.0  #set this to some small number so that it gets reset with the first image
    xmin_global = 180.0
    bins = np.arange(0.0, 180.0, stepsize)
    digit_y = np.zeros_like(bins)
    digit_norm = np.zeros_like(bins)
    for k in range(0, len(tth)):
        x = []
        y = []
        print(k)
        print(tth[k])
        filename = image_path + user + "_" + spec_name + "_scan" + str(scan_num) + "_" + str(k).zfill(4) + ".raw"
        print(filename)
        data = read_RAW(filename)
        xyz_map_prime = rotate_operation(xyz_map, tth[k])
        print(xyz_map_prime.shape)
        tth_map = cart2sphere(xyz_map_prime)
        print(tth_map.shape)
        x = tth_map.flatten()
        y = data.flatten()/i0[k]
        xmax = np.max(x)
        xmin = np.min(x)
        xmax_global = np.max([xmax, xmax_global])
        xmin_global = np.min([xmin, xmin_global])
        sort_index = np.argsort(x)
        y_0 = np.where(y < 0, np.zeros_like(y), y)
        y_1 = np.where(y < 0, np.zeros_like(y), np.ones_like(y))
        
        digit_y += np.histogram(x + stepsize, weights=y_0, range=(0,180), bins=int(math.ceil(180.0/stepsize)))[0]
        digit_norm += np.histogram(x + stepsize, weights=y_1, range=(0,180), bins=int(math.ceil(180.0/stepsize)))[0]
            
    nonzeros = np.nonzero(digit_norm)
    interp = interpolate.InterpolatedUnivariateSpline(bins[nonzeros], digit_y[nonzeros]/digit_norm[nonzeros])
    
    interpbins = np.arange(min(bins[nonzeros]), max(bins[nonzeros]), stepsize)
    interpbins = np.around(interpbins, decimals=3)
    interpy = interp(interpbins)
    
    outname = spec_path + spec_name + "_scan" + str(scan_num) + ".xye"
    outfile = open(outname, "w")
    for i in range(0, len(interpbins)):
        outfile.write(str(interpbins[i]) + "\t" + str(mult * interpy[i]) + "\t" + str(np.sqrt(mult * interpy[i])) + "\n")
    outfile.close()
    
    plt.figure()
    plt.plot(interpbins, interpy)
    
    plt.figure()
    plt.plot(bins, digit_norm)
    #plt.savefig(outname[:-4] + ".pdf")
    
plt.show()
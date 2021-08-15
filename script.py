import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
import ximu_python_library.xIMUdataClass as xIMU
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# filePath = 'datasets/straightLine'
# startTime = 6
# stopTime = 26
# samplePeriod = 1/256

# filePath = 'datasets/stairsAndCorridor'
# startTime = 5
# stopTime = 53
# samplePeriod = 1/256

# filePath = 'datasets/spiralStairs'
# startTime = 4
# stopTime = 47
# samplePeriod = 1/256

filePath = 'datasets/test'
startTime = 2
stopTime = 35
samplePeriod = 1/256

def main():
    xIMUdata = xIMU.xIMUdataClass(filePath, 'InertialMagneticSampleRate', 1/samplePeriod)
    time = xIMUdata.CalInertialAndMagneticData.Time
    

    samplePeriod_ = (time[-1] - time[0]) / (len(time) - 1)
    samplePeriod_ = round(samplePeriod_, 4)
    print("read {} imu data with cycle={}hz (calculated={}hz)".format(len(time), 1/samplePeriod, 1/samplePeriod_))

    gyrX = xIMUdata.CalInertialAndMagneticData.gyroscope[:,0] / 3.1415926 * 180
    # gyrX = gyrX - gyrX.mean()
    gyrY = xIMUdata.CalInertialAndMagneticData.gyroscope[:,1] / 3.1415926 * 180
    # gyrY = gyrY - gyrY.mean()
    gyrZ = xIMUdata.CalInertialAndMagneticData.gyroscope[:,2] / 3.1415926 * 180
    # gyrZ = gyrZ - gyrZ.mean()
        

    acc_misalignment = np.array([
        1,  -0.0020787, -0.00262097,
        0,           1,  -0.0040836,
        -0,           0,           1,
    ]).reshape([3, 3])
    acc_scale = np.array([1.00813, 1.00447, 1.00714])
    acc_bias = np.array([-0.0426319,  -0.114644, -0.0283882])

    acc = xIMUdata.CalInertialAndMagneticData.accelerometer[:, 0:3]
    acc = acc - acc.mean()
    
    acc_mean_1000 = acc[:1000].mean(axis=0)
    print("before calib: {}, {}".format(acc_mean_1000, np.linalg.norm(acc_mean_1000)))
    acc = np.matmul(acc_misalignment, (acc_scale * (acc - acc_bias)).T).T
    acc_mean_1000 = acc[:1000].mean(axis=0)
    print("before calib: {}, {}".format(acc_mean_1000, np.linalg.norm(acc_mean_1000)))
    acc = acc / 9.8

    accX = acc[:, 0]
    # accX = accX - accX.mean()
    accY = acc[:, 1]
    # accY = accY - accY.mean()
    accZ = acc[:, 2]
    # accZ = accZ - accZ.mean()


    # g = 9.81
    # accX = xIMUdata.CalInertialAndMagneticData.accelerometer[:, 0] / g
    # # accX = accX / g
    # # accX = (accX - accX.mean()) / g
    # accY = xIMUdata.CalInertialAndMagneticData.accelerometer[:, 1] / g
    # # accY = accY / g
    # accZ = xIMUdata.CalInertialAndMagneticData.accelerometer[:, 2] / g
    # accZ = accZ - accZ.mean()

    indexSel = np.all([time>=startTime,time<=stopTime], axis=0)
    time = time[indexSel]
    gyrX = gyrX[indexSel]
    gyrY = gyrY[indexSel]
    gyrZ = gyrZ[indexSel]
    accX = accX[indexSel]
    accY = accY[indexSel]
    accZ = accZ[indexSel]


    # Compute accelerometer magnitude
    acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)
    # acc_mag = np.linalg.norm(acc, axis=1)
    print("acc_mag mean: %f" % acc_mag.mean())

    # HP filter accelerometer data
    filtCutOff = 0.001
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'highpass')
    acc_magFilt = signal.filtfilt(b, a, acc_mag, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    # Compute absolute value
    acc_magFilt = np.abs(acc_magFilt)

    # LP filter accelerometer data
    filtCutOff = 5
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'lowpass')
    acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))


    # Threshold detection
    # stationary = acc_magFilt < 0.05
    stationary = np.less(acc_magFilt, 0.050)
    print("stationary rate:{}".format(1.0 * stationary.sum() / len(stationary)))

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(time,gyrX,c='r',linewidth=0.5)
    ax1.plot(time,gyrY,c='g',linewidth=0.5)
    ax1.plot(time,gyrZ,c='b',linewidth=0.5)
    ax1.set_title("gyroscope")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("angular velocity (degrees/s)")
    ax1.legend(["x","y","z"])
    ax2.plot(time,accX,c='r',linewidth=0.5)
    ax2.plot(time,accY,c='g',linewidth=0.5)
    ax2.plot(time,accZ,c='b',linewidth=0.5)
    ax2.plot(time,acc_magFilt,c='k',linestyle=":",linewidth=1)
    ax2.plot(time,stationary,c='k')
    ax2.set_title("accelerometer")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("acceleration (g)")
    ax2.legend(["x","y","z"])
    plt.show(block=False)

    # Compute orientation
    quat  = np.zeros((time.size, 4), dtype=np.float64)

    # initial convergence
    initPeriod = 2
    indexSel = time<=time[0]+initPeriod
    gyr=np.zeros(3, dtype=np.float64)
    acc = np.array([np.mean(accX[indexSel]), np.mean(accY[indexSel]), np.mean(accZ[indexSel])])
    mahony = ahrs.filters.Mahony(Kp=1, Ki=0,KpInit=1, frequency=1/samplePeriod)
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    for i in range(0, 2000):
        q = mahony.updateIMU(q, gyr=gyr, acc=acc)

    # For all data
    for t in range(0,time.size):
        if(stationary[t]):
            mahony.Kp = 0.5
        else:
            mahony.Kp = 0
        gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        acc = np.array([accX[t],accY[t],accZ[t]])
        quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)

    # -------------------------------------------------------------------------
    # Compute translational accelerations

    # Rotate body accelerations to Earth frame
    acc = []
    for x,y,z,q in zip(accX,accY,accZ,quat):
        acc.append(q_rot(np.array([x, y, z]), q_conj(q)))
    acc = np.array(acc)
    acc = acc - np.array([0,0,1])
    acc = acc * 9.81
    # print("acc:{}".format(acc))

    # Compute translational velocities
    # acc[:,2] = acc[:,2] - 9.81

    # acc_offset = np.zeros(3)
    vel = np.zeros(acc.shape)
    for t in range(1,vel.shape[0]):
        vel[t,:] = vel[t-1,:] + acc[t,:]*samplePeriod
        if stationary[t] == True:
            vel[t,:] = np.zeros(3)

    # Compute integral drift during non-stationary periods
    velDrift = np.zeros(vel.shape)
    stationaryStart = np.where(np.diff(stationary.astype(int)) == -1)[0]+1
    stationaryEnd = np.where(np.diff(stationary.astype(int)) == 1)[0]+1
    for i in range(0,stationaryStart.shape[0]):
        moo1 = vel[stationaryEnd[i]-1,:]
        moo2 = (stationaryEnd[i] - stationaryStart[i])
        driftRate = moo1 / moo2
        enum = np.arange(0,stationaryEnd[i]-stationaryStart[i])
        drift = np.array([enum*driftRate[0], enum*driftRate[1], enum*driftRate[2]]).T
        velDrift[stationaryStart[i]:stationaryEnd[i],:] = drift

    # Remove integral drift
    vel = vel - velDrift
    fig = plt.figure(figsize=(10, 5))
    plt.plot(time,vel[:,0],c='r',linewidth=0.5)
    plt.plot(time,vel[:,1],c='g',linewidth=0.5)
    plt.plot(time,vel[:,2],c='b',linewidth=0.5)
    plt.legend(["x","y","z"])
    plt.title("velocity")
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.show(block=False)

    # -------------------------------------------------------------------------
    # Compute translational position
    pos = np.zeros(vel.shape)
    for t in range(1,pos.shape[0]):
        pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod

    fig = plt.figure(figsize=(10, 5))
    plt.plot(time,pos[:,0],c='r',linewidth=0.5)
    plt.plot(time,pos[:,1],c='g',linewidth=0.5)
    plt.plot(time,pos[:,2],c='b',linewidth=0.5)
    plt.legend(["x","y","z"])
    plt.title("position")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.show(block=False)

    # -------------------------------------------------------------------------
    # Plot 3D foot trajectory

    posPlot = pos
    quatPlot = quat

    extraTime = 20
    onesVector = np.ones(int(extraTime*(1/samplePeriod)))

    # Create 6 DOF animation
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d') # Axe3D object
    ax.plot(posPlot[:,0],posPlot[:,1],posPlot[:,2])
    min_, max_ = np.min(np.min(posPlot,axis=0)), np.max(np.max(posPlot,axis=0))
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
    ax.set_zlim(min_,max_)
    ax.set_title("trajectory")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("z position (m)")
    plt.show(block=False)
    
    plt.show()

if __name__ == "__main__":
    main()

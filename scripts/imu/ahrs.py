import numpy as np

class AHRS:
    def __init__(self, **kwargs):
        self.SamplePeriod = 1 / 256
        self.Quaternion = np.array([1, 0, 0, 0])  # output quaternion describing the sensor relative to the Earth
        self.Kp = 2  # proportional gain
        self.Ki = 0  # integral gain
        self.KpInit = 200  # proportional gain used during initialization
        self.InitPeriod = 5  # initialization period in seconds

        for key, value in kwargs.items():
            if key == 'SamplePeriod':
                self.SamplePeriod = value
            elif key == 'Quaternion':
                self.Quaternion = value
                self.q = self.quaternConj(self.Quaternion)
            elif key == 'Kp':
                self.Kp = value
            elif key == 'Ki':
                self.Ki = value
            elif key == 'KpInit':
                self.KpInit = value
            elif key == 'InitPeriod':
                self.InitPeriod = value
            else:
                raise ValueError('Invalid argument')

        self.q = np.array([1, 0, 0, 0])  # internal quaternion describing the Earth relative to the sensor
        self.IntError = np.array([0, 0, 0])  # integral error
        self.KpRamped = self.KpInit  # internal proportional gain used to ramp during initialization


    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        raise NotImplementedError('This method is unimplemented')

    def UpdateIMU(self, Gyroscope, Accelerometer):
        # Normalise accelerometer measurement
        norm_accel = np.linalg.norm(Accelerometer)
        if norm_accel == 0:
            raise Warning('Accelerometer magnitude is zero. Algorithm update aborted.')
        else:
            Accelerometer /= norm_accel

        # Compute error between estimated and measured direction of gravity
        v = np.array([2 * (self.q[1] * self.q[3] - self.q[0] * self.q[2]),
                      2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]),
                      self.q[0] ** 2 - self.q[1] ** 2 - self.q[2] ** 2 + self.q[3] ** 2])
        error = np.cross(v, Accelerometer)

        # Compute ramped Kp value used during init period
        # if self.KpRamped > self.Kp:
        #     self.IntError = np.array([0, 0, 0])
        #     self.KpRamped -= (self.KpInit - self.Kp) / (self.InitPeriod / self.SamplePeriod)
        # else:  # init period complete
        #     self.KpRamped = self.Kp
        #     self.IntError = self.IntError + error  # compute integral feedback terms (only outside of init period)

        # Apply feedback terms
        Ref = Gyroscope - (self.Kp * error + self.Ki * self.IntError)

        # Compute rate of change of quaternion
        pDot = 0.5 * self.quaternProd(self.q, np.array([0, Ref[0], Ref[1], Ref[2]]))
        self.q = self.q + pDot * self.SamplePeriod  # integrate rate of change of quaternion
        self.q /= np.linalg.norm(self.q)  # normalise quaternion

        # Store conjugate
        self.Quaternion = self.quaternConj(self.q)

    def Reset(self):
        self.KpRamped = self.KpInit  # start Kp ramp-down
        self.IntError = np.array([0, 0, 0])  # reset integral terms
        self.q = np.array([1, 0, 0, 0])  # set quaternion to alignment

    def set_Quaternion(self, value):
        if np.linalg.norm(value) == 0:
            raise ValueError('Quaternion magnitude cannot be zero.')
        value /= np.linalg.norm(value)
        self.Quaternion = value
        self.q = self.quaternConj(value)

    def quaternProd(self, a, b):
        ab = np.zeros(4)
        ab[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        ab[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        ab[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        ab[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return ab

    def quaternConj(self, q):
        qConj = np.array([q[0], -q[1], -q[2], -q[3]])
        return qConj
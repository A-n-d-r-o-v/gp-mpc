
class State():
    def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0,
                 roll=0.0, pitch=0.0, yaw=0.0, roll_ref=0.0, pitch_ref=0.0,
                 thrust=9.806, roll_ref_dot=0.0, pitch_ref_dot=0.0, thrust_dot=0.0, time=0.0):
        # Position
        self.x = x
        self.y = y
        self.z = z
        # Velocity
        self.vx= vx
        self.vy= vy
        self.vz= vz
        # Orientation
        self.roll  = roll
        self.pitch = pitch
        self.yaw   = yaw
        # Controls
        self.roll_ref  = roll_ref
        self.pitch_ref = pitch_ref
        self.thrust    = thrust
        # help variables
        self.roll_ref_dot  = roll_ref_dot
        self.pitch_ref_dot = pitch_ref_dot
        self.thrust_dot    = thrust_dot
        self.time          = time

    def getXYZVXVYVZStr(self):
        return f"{self.x:.6f} {self.y:.6f} {self.z:.6f} {self.vx:.6f} {self.vy:.6f} {self.vz:.6f}"

    def __str__(self):
        return f"x: {self.x} y: {self.y} z: {self.z} vx: {self.vx} vy: {self.vy} vz: {self.vz} r: {self.roll} p: {self.pitch} yaw: {self.yaw} thrust: {self.thrust}"



import mujoco as mj
import numpy as np
from numpy.random import uniform, randn
from numpy.linalg import norm
from numpy import cos, sin


class EnvRandomizer(object):
    def __init__(self, model: mj.MjModel):
        self.model = model
        # Number of bodies and geoms
        self.nbody = model.nbody
        self.ngeom = model.ngeom
        self.nu = model.nu

        # Default inertia properties
        self._default_body_ipos = model.body_ipos.copy()  # local position of center of mass (nbody x 3)
        self._default_body_iquat = model.body_iquat.copy()  # local orientation of inertia ellipsoid (nbody x 4)
        self._default_body_mass = model.body_mass.copy()  # mass (nbody x 1)
        self._default_body_inertia = model.body_inertia.copy()  # diagonal inertia in ipos/iquat frame (nbody x 3)

        # Default actuator properties
        self._default_actuator_gear = model.actuator_gear.copy()

        # Noise scales
        self.ipos_noise_scale = 0.01  # m
        self.iquat_noise_scale = 5  # deg
        self.mass_noise_scale = 0.05
        self.inertia_noise_scale = 0.05
        self.actuator_gear_noise_scale = 0.05

    def randomize_env(self):
        for i in range(self.nbody):
            self.model.body_ipos[i] = self._default_body_ipos[i] + uniform(size=3, low=-self.ipos_noise_scale, high=self.ipos_noise_scale)
            body_iquat = random_deviation_quaternion(self._default_body_iquat[i], self.iquat_noise_scale)
            self.model.body_iquat[i] = body_iquat / norm(body_iquat)  # normalize quaternion
            self.model.body_mass[i] = self._default_body_mass[i] * (1.0 + uniform(low=-self.mass_noise_scale, high=self.mass_noise_scale))
            self.model.body_inertia[i] = self._default_body_inertia[i] * (1.0 + uniform(size=3, low=-self.inertia_noise_scale, high=self.inertia_noise_scale))
        
        for gear in self.model.actuator_gear:
            gear *= (1.0 + uniform(low=-self.actuator_gear_noise_scale, high=self.actuator_gear_noise_scale))

    def reset(self):
        self.model.body_ipos = self._default_body_ipos
        self.model.body_iquat = self._default_body_iquat
        self.model.body_mass = self._default_body_mass
        self.model.body_inertia = self._default_body_inertia
        self.model.actuator_gear = self._default_actuator_gear

def random_deviation_quaternion(original_quaternion, max_angle_degrees):
    random_axis = randn(3)
    random_axis /= norm(random_axis)
    random_angle = uniform(low=0, high=max_angle_degrees) * np.pi / 180
    w = cos(random_angle / 2)
    xyz = random_axis * sin(random_angle / 2)
    random_quaternion = np.concatenate([[w], xyz])
    return quaternion_multiply(original_quaternion, random_quaternion)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


if __name__ == "__main__":
    xml_file = "../../assets/quadrotor_falcon.xml"
    model = mj.MjModel.from_xml_path(xml_file)
    env_randomizer = EnvRandomizer(model=model)
    env_randomizer.randomize_env()

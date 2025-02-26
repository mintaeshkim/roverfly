import mujoco as mj
import numpy as np
from numpy.random import uniform, randn
from numpy.linalg import norm
from numpy import cos, sin
from copy import copy
from rotation_transformations import *


class EnvRandomizer(object):
    def __init__(self, model: mj.MjModel):
        self.model = model
        # Number of bodies and geoms
        self.nbody = self.model.nbody
        self.ngeom = self.model.ngeom
        self.nu = self.model.nu

        # Default inertia and gear properties
        self._default_body_ipos = copy(self.model.body_ipos)  # local position of center of mass (nbody x 3)
        self._default_body_iquat = copy(self.model.body_iquat)  # local orientation of inertia ellipsoid (nbody x 4)
        self._default_body_mass = copy(self.model.body_mass)  # mass (nbody x 1)
        self._default_body_inertia = copy(self.model.body_inertia)  # diagonal inertia in ipos/iquat frame (nbody x 3)
        self._default_actuator_gear = copy(self.model.actuator_gear)  # Default actuator properties

        # Default noise scales
        self._default_ipos_noise_scale = 0  # m
        self._default_iquat_noise_scale = 0  # deg
        self._default_mass_noise_scale = 0.1
        self._default_inertia_noise_scale = 0.1
        self._default_actuator_gear_noise_scale = 0.1

        # Noise scales
        self.reset_noise_scale()

    def randomize_env(self, model):
        model = self.reset_env(model=model)
        
        for i in range(self.nbody):
            model.body_ipos[i] = self._default_body_ipos[i] + uniform(size=3, low=-self.ipos_noise_scale, high=self.ipos_noise_scale)
            body_iquat = random_deviation_quaternion(self._default_body_iquat[i], self.iquat_noise_scale)
            model.body_iquat[i] = body_iquat / norm(body_iquat)  # normalize quaternion
            model.body_mass[i] = self._default_body_mass[i] * (1.0 + uniform(low=-self.mass_noise_scale, high=self.mass_noise_scale))
            model.body_inertia[i] = self._default_body_inertia[i] * (1.0 + uniform(size=3, low=-self.inertia_noise_scale, high=self.inertia_noise_scale))
        
        for gear in model.actuator_gear:
            gear *= 1.0 + uniform(low=-self.actuator_gear_noise_scale, high=self.actuator_gear_noise_scale, size=len(gear))

        # print("body_ipos: \n", model.body_ipos[1])
        # print("body_iquat: \n", model.body_iquat[1])
        # print("body_mass: \n", model.body_mass[1])
        # print("body_inertia: \n", model.body_inertia[1])
        # R = quat2rot(model.body_iquat[1])
        # print("full_inertia: \n", R @ np.diag(model.body_inertia[1]) @ R.T)
        # print("actuator_gear: \n", model.actuator_gear)

        return model

    def reset_env(self, model):
        model.body_ipos = self._default_body_ipos
        model.body_iquat = self._default_body_iquat
        model.body_mass = self._default_body_mass
        model.body_inertia = self._default_body_inertia
        model.actuator_gear = self._default_actuator_gear
        return model

    def set_noise_scale(self, progress):
        self.reset_noise_scale()
        self.ipos_noise_scale = self._default_ipos_noise_scale * progress  # [0, 0.005]
        self.iquat_noise_scale = self._default_iquat_noise_scale * progress  # [0, 5]
        self.mass_noise_scale = self._default_mass_noise_scale * progress  # [0, 0.1]
        self.inertia_noise_scale = self._default_inertia_noise_scale * progress  # [0, 0.1]
        self.actuator_gear_noise_scale = self._default_actuator_gear_noise_scale * progress  # [0, 0.1]
        
        # print(f"ipos_noise_scale: {self.ipos_noise_scale}")
        # print(f"iquat_noise_scale: {self.iquat_noise_scale}")
        # print(f"mass_noise_scale: {self.mass_noise_scale}")
        # print(f"inertia_noise_scale: {self.inertia_noise_scale}")
        # print(f"actuator_gear_noise_scale: {self.actuator_gear_noise_scale}")

    def reset_noise_scale(self):
        self.ipos_noise_scale = self._default_ipos_noise_scale  # m
        self.iquat_noise_scale = self._default_iquat_noise_scale  # deg
        self.mass_noise_scale = self._default_mass_noise_scale
        self.inertia_noise_scale = self._default_inertia_noise_scale
        self.actuator_gear_noise_scale = self._default_actuator_gear_noise_scale

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
    xml_file = "../../assets/quadrotor_mini.xml"
    model = mj.MjModel.from_xml_path(xml_file)
    env_randomizer = EnvRandomizer(model=model)
    for _ in range(1):
        env_randomizer.set_noise_scale(progress=0.2)
        env_randomizer.randomize_env(model=model)
        print()

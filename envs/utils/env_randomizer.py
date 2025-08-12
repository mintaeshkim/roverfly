import mujoco as mj
import numpy as np
from numpy.random import choice, uniform, randn
from numpy.linalg import norm
from numpy import cos, sin, zeros
from copy import copy
from rotation_transformations import *


class EnvRandomizer(object):
    def __init__(self, model: mj.MjModel):
        self.model = model
        self.has_payload = "payload" in [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        
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
        
        # Noise scales
        self.ipos_noise_scale = 0.025  # m
        self.iquat_noise_scale = 5  # deg
        self.mass_noise_scale = 0.1
        self.inertia_noise_scale = 0.1
        self.actuator_gear_noise_scale = 0.1

        # Payload and tendon
        if self.has_payload:
            # Payload ID
            self.payload_body_id = self.model.body(name="payload").id
            self.payload_site_id = self.model.site(name="hook_payload_site").id
            self.core_site_id = self.model.site(name="hook_core_site").id

            # Default payload and tendon properties
            self._default_payload_pos = zeros(3)
            self._default_hook_core_site_pos = np.array([0, 0, 1])
            self._default_hook_payload_site_pos = zeros(3)
            self._default_payload_mass = 0.1
            self._default_tendon_max_length = 1.0

            # Payload and tendon scales
            self.payload_mass_scale = 2.0  # [0.0, 0.2] kg
            self.tendon_length_scale = 1.0  # [0.025, 1.0] m

            # Rotor dynamics
            self._default_tau_up = 0.2164
            self._default_tau_down = 0.1644

    def randomize_env(self, model):
        model = self.reset_env(model=model)
        
        for i in range(self.nbody):
            if self.has_payload and not i == self.payload_body_id:
                model.body_ipos[i] = self._default_body_ipos[i] + uniform(size=3, low=-self.ipos_noise_scale, high=self.ipos_noise_scale)
                body_iquat = random_deviation_quaternion(self._default_body_iquat[i], self.iquat_noise_scale)
                model.body_iquat[i] = body_iquat / norm(body_iquat)  # normalize quaternion
                model.body_mass[i] = self._default_body_mass[i] * (1.0 + uniform(low=-self.mass_noise_scale, high=self.mass_noise_scale))
                model.body_inertia[i] = self._default_body_inertia[i] * (1.0 + uniform(size=3, low=-self.inertia_noise_scale, high=self.inertia_noise_scale))
        
        for gear in model.actuator_gear:
            gear *= 1.0 + uniform(low=-self.actuator_gear_noise_scale, high=self.actuator_gear_noise_scale, size=len(gear))
        
        if self.has_payload:
            payload_mass = self._default_body_mass[self.payload_body_id] * uniform(low=0, high=self.payload_mass_scale)
            tendon_length = uniform(low=0, high=1)
            # payload_mass = 0.1
            # tendon_length = 0.5
            model.body_ipos[self.payload_body_id] = self._default_hook_core_site_pos + np.array([0, 0, -tendon_length])
            model.body_mass[self.payload_body_id] = payload_mass
            model.tendon_range[0][1] = tendon_length

        tau_up = self._default_tau_up * uniform(0.05, 0.2)
        tau_down = self._default_tau_down * uniform(0.05, 0.2)

        # print("body_ipos: \n", model.body_ipos)
        # print("body_iquat: \n", model.body_iquat)
        # print("body_mass: \n", model.body_mass)
        # print("body_inertia: \n", model.body_inertia)
        # print("actuator_gear: \n", model.actuator_gear)
        # if self.has_payload:
        #     print("hook_core_site_pos: \n", self._default_hook_core_site_pos)
        #     print("hook_payload_site_pos: \n", self._default_hook_payload_site_pos)
        #     print("payload_pos: \n", model.body_ipos[self.payload_body_id])
        #     print("payload_mass: \n", model.body_mass[self.payload_body_id])
        #     print("tendon_range: \n", model.tendon_range[0])
        # print()
        
        return model, payload_mass, tendon_length, tau_up, tau_down

    def reset_env(self, model):
        model.body_ipos = self._default_body_ipos
        model.body_iquat = self._default_body_iquat
        model.body_mass = self._default_body_mass
        model.body_inertia = self._default_body_inertia
        model.actuator_gear = self._default_actuator_gear
        if self.has_payload:
            model.body_ipos[self.payload_body_id] = self._default_payload_pos
            model.site_pos[self.payload_site_id] = self._default_hook_payload_site_pos
            model.body_mass[self.payload_body_id] = self._default_payload_mass
            model.tendon_range[0][1] = self._default_tendon_max_length

        return model

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
    xml_file = "../../assets/quadrotor_falcon_payload.xml"
    model = mj.MjModel.from_xml_path(xml_file)
    env_randomizer = EnvRandomizer(model=model)
    for _ in range(5):
        env_randomizer.randomize_env(model=model)
    
    xml_file = "../../assets/quadrotor_falcon.xml"
    model = mj.MjModel.from_xml_path(xml_file)
    env_randomizer = EnvRandomizer(model=model)
    for _ in range(5):
        env_randomizer.randomize_env(model=model)
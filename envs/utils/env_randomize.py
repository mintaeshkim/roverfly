import numpy as np
import xml.etree.ElementTree as ET

def randomize_mujoco_parameters(original_xml_path, modified_xml_path, mass_range):
    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    total_original_mass = 0

    # Randomize mass
    for body in root.findall(".//body"):
        if str(body.get('name')) == "quadrotor":
            print(body.get('name'))
            for geom in body.findall(".//geom"):
                original_mass = float(geom.get("mass"))
                total_original_mass += original_mass
                print(original_mass)
                randomized_mass = np.round(original_mass * (1 + np.random.uniform(*mass_range)), 2)
                geom.set('mass', str(randomized_mass))

    tree.write(modified_xml_path)
    print(np.round(total_original_mass, 2))

# Example usage:
randomize_mujoco_parameters(original_xml_path="../assets/quadrotor_x_cfg.xml",
                            modified_xml_path="../assets/quadrotor_x_cfg_modified.xml", 
                            mass_range=(0, 0.2))

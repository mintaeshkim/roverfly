import mujoco as mj
import numpy as np


def get_sensor_data_by_name(model, data, sensor_name):
    """
    Retrieve sensor data using the sensor's name.

    :param model: MuJoCo model instance
    :param data: MuJoCo data instance
    :param sensor_name: Name of the sensor
    :return: Sensor data (if found), None otherwise
    """
    try:
        # Get the sensor ID by name
        sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)

        # Get the address of the sensor data
        sensor_adr = model.sensor_adr[sensor_id]

        # Get the dimension of the sensor (how many values it reports, e.g., 3 for an accelerometer)
        sensor_dim = model.sensor_dim[sensor_id]

        # Retrieve the sensor data (it might span multiple entries if it's multi-dimensional)
        sensor_data = data.sensordata[sensor_adr:sensor_adr + sensor_dim].copy()

        return sensor_data

    except Exception as e:
        print(f"Error retrieving sensor data for '{sensor_name}': {e}")
        return None


def get_bodyIDs(model, body_list):
    bodyID_dic = {}
    jntID_b_dic = {}
    qposID_b_dic = {}
    qvelID_b_dic = {}
    for bodyName in body_list:
        mjID = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bodyName)
        jntID = model.body_jntadr[mjID]  # joint ID
        jvelID = model.body_dofadr[mjID]  # joint velocity
        posID = model.jnt_qposadr[jntID]  # joint position
        bodyID_dic[bodyName] = mjID
        jntID_b_dic[bodyName] = jntID
        qposID_b_dic[bodyName] = posID
        qvelID_b_dic[bodyName] = jvelID
    return bodyID_dic, jntID_b_dic, qposID_b_dic, qvelID_b_dic


def get_jntIDs(model, jnt_list):
    jointID_dic = {}
    jointPosID_dic = {}
    for jointName in jnt_list:
        jointID = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jointName)
        jointID_dic[jointName] = jointID
        jointPosID_dic[jointName] = model.jnt_qposadr[jointID]
    return jointID_dic, jointPosID_dic


def local2global(data, model, local_point, body):
    """
    Convert a position from the local frame of a body to the global frame.

    :param data: MuJoCo data
    :param local_pos: Position in the local frame (3x1 numpy array)
    :param body: Name of the body
    :return: Position in the global frame (3x1 numpy array)
    """
    # Get body ID
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    global_point = np.array(data.xpos[body_id]).reshape(3, 1) + data.xmat[body_id].reshape(3, 3) @ np.array(local_point).reshape(3, 1)
    # data.xpos[body_id] is identical to data.xipos[body_id] (position in global frame)
    return global_point.reshape(1,3)


def vec2rot(t,f=[0,0,1]):
    # a and b are in the form of numpy array
    if np.linalg.norm(t) < 1e-8:  # t is too small to be normalized
        return np.eye(3)
    t = np.array(t).flatten() / np.linalg.norm(t)
    v = np.cross(f, t)
    if np.linalg.norm(v) == 0:
        return np.eye(3)
    u = v / np.linalg.norm(v)
    c = np.dot(f, t)

    if c**2 == 1:
        return np.eye(3)

    h = (1 - c) / (1 - c ** 2)
    vx, vy, vz = v
    rot = np.array([[c + h * vx ** 2, h * vx * vy - vz, h * vx * vz + vy],
                   [h * vx * vy + vz, c + h * vy ** 2, h * vy * vz - vx],
                   [h * vx * vz - vy, h * vy * vz + vx, c + h * vz ** 2]])

    return rot


def draw_line(viewer, start, end, width=0.1, rgba=[1, 0, 0, 1]):
    """
    Draw a line in the viewer.

    :param model: MuJoCo model
    :param data: MuJoCo data
    :param start: Start position of the line (3x1 numpy array)
    :param end: End position of the line (3x1 numpy array)
    :param rgba: Color of the line (4-element list)
    :param viewer: MuJoCo viewer
    """
    start = np.array(start).flatten()
    end = np.array(end).flatten()
    length = np.linalg.norm(np.array(end-start))
    if length < 1e-4:  # ignore coincident points
        # print("Coincident points or Line length is too small to be visualized.")
        return
    viewer.add_marker(pos=start, type=mj.mjtGeom.mjGEOM_LINE,
                      mat=vec2rot(np.array(end-start)/ length),
                      size=[width,width,length], rgba=rgba)


def draw_curve(viewer, points, width, rgba=[1, 0, 0, 1], body=None, model=None, data=None):
    """
    Draw a curve in the viewer.
    points: list of 3x1 numpy array
    """
    if body is not None:
        if (model is None) or (data is None):
            print("Model and data must be provided to convert local points to global points.")
            raise ValueError("Model and data must be provided to convert local points to global points.")
        points = [local2global(data, model, p, body) for p in points]
    for i in range(len(points) - 1):
        draw_line(viewer, points[i], points[i + 1], width, rgba)


def draw_arrow(viewer, start, end, width=0.01, rgba=[1, 0, 0, 1], body=None, model=None, data=None, label=None, **kwargs):
    if body is not None:
        if (model is None) or (data is None):
            print("Model and data must be provided to convert local points to global points.")
            raise ValueError("Model and data must be provided to convert local points to global points.")
        start = local2global(data, model, start, body)
        end = local2global(data, model, end, body)
    marker_params = {
        "pos": start,
        "type": mj.mjtGeom.mjGEOM_ARROW,
        "mat": vec2rot(np.array(end-start)),
        "size": [width, width, 2*np.linalg.norm(np.array(end-start))],
        "rgba": rgba,
        **kwargs
    }
    if label:
        marker_params["label"] = label
    viewer.add_marker(**marker_params)


def draw_force(viewer, pos, force, scale=0.5, width=0.01, rgba=[1, 0, 0, 1], body=None, model=None, data=None, label=None, **kwargs):
    force_norm = np.linalg.norm(force)
    if viewer is not None and force_norm > 0:
        draw_arrow(viewer, pos, pos + scale*force, width=width, rgba=rgba, body=body, model=model, data=data, label="force: %.2f N" % force_norm)


def draw_torque(viewer, pos, torque, scale=0.5, rgba=[0, 1, 0, 1], body=None, model=None, data=None, label=None, **kwargs):
    torque_norm = np.linalg.norm(torque)
    print('torque:',torque)
    if viewer is not None and torque_norm > 1e-8:
        draw_arrow(viewer, pos, pos + scale*torque, width=0.01, rgba=rgba, body=body, model=model, data=data, label="torque: %.2f N-m" % torque_norm)
        # torque_axis = np.squeeze(torque / torque_norm)
        # print('torque_axis:',torque_axis)
        # # Create a circular arrow to represent torque
        # circle_points = 10
        # radius = scale * torque_norm
        # for i in range(circle_points):
        #     angle1 = 2 * np.pi * i / circle_points
        #     angle2 = 2 * np.pi * (i + 1) / circle_points
        #     p1 = np.array(pos).reshape(3,1) + radius * (vec2rot(torque_axis) @ np.array([np.cos(angle1), np.sin(angle1), 0]).reshape(3, 1))
        #     p2 = np.array(pos).reshape(3,1) + radius * (vec2rot(torque_axis) @ np.array([np.cos(angle2), np.sin(angle2), 0]).reshape(3, 1))
        #     if i == 0:
        #         viewer.add_marker(pos=p1, type=mj.mjtGeom.mjGEOM_ARROW, mat=vec2rot(np.squeeze(p2 - p1)),
        #                           size=[0.05 * radius, 0.05 * radius, 2 * 2 * np.pi * radius / circle_points],
        #                           rgba=rgba, label="torque: %.2f N-m" % torque_norm)
        #     else:
        #         viewer.add_marker(pos=p1, type=mj.mjtGeom.mjGEOM_ARROW, mat=vec2rot(np.squeeze(p2 - p1)),
        #                           size=[0.05 * radius, 0.05 * radius, 2 * 2 * np.pi * radius / circle_points],
        #                           rgba=rgba)


def get_cartesian_fluid_FT(m, d, body_ids):
    """
    Returns the fluid forces in Cartesian space for specified bodies.

    Parameters:
    m : mjModel The MuJoCo model.
    d : mjData  The MuJoCo data containing the simulation state.
    body_ids : int or list of ints
        The body ID or list of body IDs for which to compute Cartesian forces.

    Returns:
    cartesian_forces : dict
        A dictionary where keys are body IDs and values are the Cartesian forces for each body.
    """
    if isinstance(body_ids, int):
        body_ids = [body_ids]  # Convert to list if a single body ID is given

    cartesian_forces = {}

    j_list = []
    for body_id in body_ids:
        # Ensure body_id is valid
        if body_id < 0 or body_id >= m.nbody:
            raise ValueError(f"Invalid body_id {body_id}, it must be in the range 0 to {m.nbody - 1}")

        # Compute the Jacobian for the current body
        jacp = np.zeros((3, m.nv))  # Translational part of the Jacobian
        jacr = np.zeros((3, m.nv))  # Rotational part of the Jacobian

        # Compute the Jacobian for the given body (position part)
        mj.mj_jacBody(m, d, jacp, jacr, body_id)

        j_list.append(jacp)
        # Compute Cartesian forces (using the translational part of the Jacobian)
        # cartesian_force_body = np.dot(np.linalg.pinv(jacp.T), qfrc_fluid_body)
        # cartesian_torque_body = np.dot(np.linalg.pinv(jacr.T), qfrc_fluid_body)

        # Store the force in a dictionary with body ID as the key
        # cartesian_forces[body_id] = np.array([cartesian_force_body[:], cartesian_torque_body[:]]).flatten()
    # print('j_list:',np.vstack(j_list))
    return cartesian_forces.copy()


def draw_base_fluid_force(model, data, viewer):
    # print('qfrc_fluid:',data.qfrc_fluid)
    force = data.qfrc_fluid[0:3]
    torque = data.qfrc_fluid[3:6]
    # print('force:',force)
    # print('torque:',torque)
    draw_force(viewer, data.sensordata[0:3], force, width=0.01, rgba=[1, 0, 0, 1])
    draw_torque(viewer, data.sensordata[0:3], torque, width=0.01, rgba=[0, 1, 0, 1])


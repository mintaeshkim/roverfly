import mujoco as mj


model = mj.MjModel.from_xml_path("../../assets/quadrotor_falcon_payload.xml")
# model = mj.MjModel.from_xml_path("../../assets/quadrotor_x_cfg_payload_nominal.xml")
data = mj.MjData(model)

def get_body_indices(model, body_name):
    """Get the qpos and qvel indices for a given body."""
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    joint_id = model.body_jntadr[body_id]
    qposadr = model.jnt_qposadr[joint_id]
    qveladr = model.jnt_dofadr[joint_id]
    return qposadr, qveladr

def print_body_state(data, model, body_name):
    """Print the position and velocity of a given body."""
    qposadr, qveladr = get_body_indices(model, body_name)
    print(qposadr, qveladr)
    qpos = data.qpos[qposadr:qposadr+7]  # For free joints, 7 elements (3 pos + 4 quat)
    qvel = data.qvel[qveladr:qveladr+6]  # For free joints, 6 elements (3 linear vel + 3 angular vel)
    print(f"Body: {body_name}")
    print(f"Position: {qpos[:3]}")
    print(f"Orientation: {qpos[3:]}")
    print(f"Linear Velocity: {qvel[:3]}")
    print(f"Angular Velocity: {qvel[3:]}")


# print_body_state(data, model, 'quadrotor_0')    # pos = qpos[0:3], ori = qpos[3:7], vel = qvel[0:3], angvel = qvel[3:6]
# print_body_state(data, model, 'quadrotor_1')    # pos = qpos[11:14], ori = qpos[14:18], vel = qvel[9:12], angvel = qvel[12:15]
# print_body_state(data, model, 'payload')        # pos = qpos[22:25], ori = qpos[25:29], vel = qvel[18:21], angvel = qvel[21:24]
# print_body_state(data, model, 'hook_core_0')    # ori = qpos[7:11], angvel = qvel[6:9]
# print_body_state(data, model, 'hook_core_1')    # ori = qpos[18:22], angvel = qvel[15:18]
# print_body_state(data, model, 'hook_payload')   # ori = qpos[29:33], angvel = qvel[24:27]

print_body_state(data, model, 'quadrotor')
print_body_state(data, model, 'hook_core')
print_body_state(data, model, 'payload')
print_body_state(data, model, 'hook_payload')

print(data.qpos)
print(data.qvel)
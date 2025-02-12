import mujoco as mj

def get_bodyIDs(body_list):
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

xml_file = "../assets/quadrotor_x_cfg.xml"
model = mj.MjModel.from_xml_path(xml_file)

body_list = ["quadrotor"]
bodyID_dic, jntID_b_dic, qposID_b_dic, qvelID_b_dic = get_bodyIDs(body_list)

print(model.body_inertia[bodyID_dic["quadrotor"]])
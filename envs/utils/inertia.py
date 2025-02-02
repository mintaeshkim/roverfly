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
# print(model.body_inertia[bodyID_dic["arm00"]])


# import mujoco

# # Load the MuJoCo model and data from the XML file
# model = mujoco.MjModel.from_xml_path("../assets/quadrotor_x_cfg.xml")
# data = mujoco.MjData(model)

# # Function to calculate the total inertia for a list of body names
# def calculate_total_inertia(body_names):
#     total_inertia = [0.0, 0.0, 0.0]  # Initialize total inertia as zero for each axis

#     for body_name in body_names:
#         # Get the body ID for the current body name
#         body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
#         # Extract the inertia for the current body
#         body_inertia = model.body_inertia[body_id]
        
#         # Sum the inertia of this body to the total inertia
#         total_inertia[0] += body_inertia[0]  # Ixx
#         total_inertia[1] += body_inertia[1]  # Iyy
#         total_inertia[2] += body_inertia[2]  # Izz

#     return total_inertia

# # List of all body names in the quadrotor system
# body_names = [
#     "core", "arm00", "arm10", "arm20", "arm30",
#     "arm01", "thruster0", "arm11", "thruster1", 
#     "arm21", "thruster2", "arm31", "thruster3"
# ]

# # Calculate the total inertia of the quadrotor system
# total_inertia = calculate_total_inertia(body_names)

# # Print the total inertia of the quadrotor system
# print(f"Total Inertia of Quadrotor (Ixx, Iyy, Izz): {total_inertia}")

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Armature values (estimated based on effort limits: 164Nm -> larger motor, 61Nm -> smaller motor)
ARMATURE_7520_22 = 0.025101925  # for 164Nm joints (hip/knee)
ARMATURE_7520_14 = 0.010177520  # for 61Nm joints (ankle/waist/arms)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2

DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ

PM01_EDU_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/pm01_edu/urdf/serial_pm01_edu.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            # Legs - neutral standing pose
            "J00_HIP_PITCH_.*": -0.1,
            "J01_HIP_ROLL_.*": 0.0,
            "J02_HIP_YAW_.*": 0.0,
            "J03_KNEE_PITCH_.*": 0.2,
            "J04_ANKLE_PITCH_.*": -0.1,
            "J05_ANKLE_ROLL_.*": 0.0,
            # Waist
            "J12_WAIST_YAW": 0.0,
            # Arms - slightly out to sides
            "J13_SHOULDER_PITCH_.*": 0.3,
            "J14_SHOULDER_ROLL_.*": 0.0,
            "J15_SHOULDER_YAW_.*": 0.0,
            "J16_ELBOW_PITCH_.*": 0.5,
            "J17_ELBOW_YAW_.*": 0.0,
            # Head
            "J23_HEAD_YAW": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "J00_HIP_PITCH_.*",
                "J01_HIP_ROLL_.*",
                "J02_HIP_YAW_.*",
                "J03_KNEE_PITCH_.*",
            ],
            effort_limit_sim={
                "J00_HIP_PITCH_.*": 164.0,
                "J01_HIP_ROLL_.*": 164.0,
                "J02_HIP_YAW_.*": 61.0,
                "J03_KNEE_PITCH_.*": 164.0,
            },
            velocity_limit_sim={
                "J00_HIP_PITCH_.*": 26.3,
                "J01_HIP_ROLL_.*": 26.3,
                "J02_HIP_YAW_.*": 35.2,
                "J03_KNEE_PITCH_.*": 26.3,
            },
            stiffness={
                "J00_HIP_PITCH_.*": STIFFNESS_7520_22,
                "J01_HIP_ROLL_.*": STIFFNESS_7520_22,
                "J02_HIP_YAW_.*": STIFFNESS_7520_14,
                "J03_KNEE_PITCH_.*": STIFFNESS_7520_22,
            },
            damping={
                "J00_HIP_PITCH_.*": DAMPING_7520_22,
                "J01_HIP_ROLL_.*": DAMPING_7520_22,
                "J02_HIP_YAW_.*": DAMPING_7520_14,
                "J03_KNEE_PITCH_.*": DAMPING_7520_22,
            },
            armature={
                "J00_HIP_PITCH_.*": ARMATURE_7520_22,
                "J01_HIP_ROLL_.*": ARMATURE_7520_22,
                "J02_HIP_YAW_.*": ARMATURE_7520_14,
                "J03_KNEE_PITCH_.*": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "J04_ANKLE_PITCH_.*",
                "J05_ANKLE_ROLL_.*",
            ],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J12_WAIST_YAW"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "J13_SHOULDER_PITCH_.*",
                "J14_SHOULDER_ROLL_.*",
                "J15_SHOULDER_YAW_.*",
                "J16_ELBOW_PITCH_.*",
                "J17_ELBOW_YAW_.*",
                "J18_SHOULDER_PITCH_.*",
                "J19_SHOULDER_ROLL_.*",
                "J20_SHOULDER_YAW_.*",
                "J21_ELBOW_PITCH_.*",
                "J22_ELBOW_YAW_.*",
                "J23_HEAD_YAW",
            ],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
    },
)

PM01_EDU_ACTION_SCALE = {}
for a in PM01_EDU_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PM01_EDU_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

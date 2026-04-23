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
            ".*_HIP_PITCH_.*": -0.1,
            ".*_HIP_ROLL_.*": 0.0,
            ".*_HIP_YAW_.*": 0.0,
            ".*_KNEE_PITCH_.*": 0.2,
            ".*_ANKLE_PITCH_.*": -0.1,
            ".*_ANKLE_ROLL_.*": 0.0,
            # Waist
            ".*_WAIST_YAW": 0.0,
            # Arms - slightly out to sides
            ".*_SHOULDER_PITCH_.*": 0.3,
            ".*_SHOULDER_ROLL_.*": 0.0,
            ".*_SHOULDER_YAW_.*": 0.0,
            ".*_ELBOW_PITCH_.*": 0.5,
            ".*_ELBOW_YAW_.*": 0.0,
            # Head
            ".*_HEAD_YAW": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_HIP_PITCH_.*",
                ".*_HIP_ROLL_.*",
                ".*_HIP_YAW_.*",
                ".*_KNEE_PITCH_.*",
            ],
            effort_limit_sim={
                ".*_HIP_PITCH_.*": 164.0,
                ".*_HIP_ROLL_.*": 164.0,
                ".*_HIP_YAW_.*": 61.0,
                ".*_KNEE_PITCH_.*": 164.0,
            },
            velocity_limit_sim={
                ".*_HIP_PITCH_.*": 26.3,
                ".*_HIP_ROLL_.*": 26.3,
                ".*_HIP_YAW_.*": 35.2,
                ".*_KNEE_PITCH_.*": 26.3,
            },
            stiffness={
                ".*_HIP_PITCH_.*": STIFFNESS_7520_22,
                ".*_HIP_ROLL_.*": STIFFNESS_7520_22,
                ".*_HIP_YAW_.*": STIFFNESS_7520_14,
                ".*_KNEE_PITCH_.*": STIFFNESS_7520_22,
            },
            damping={
                ".*_HIP_PITCH_.*": DAMPING_7520_22,
                ".*_HIP_ROLL_.*": DAMPING_7520_22,
                ".*_HIP_YAW_.*": DAMPING_7520_14,
                ".*_KNEE_PITCH_.*": DAMPING_7520_22,
            },
            armature={
                ".*_HIP_PITCH_.*": ARMATURE_7520_22,
                ".*_HIP_ROLL_.*": ARMATURE_7520_22,
                ".*_HIP_YAW_.*": ARMATURE_7520_14,
                ".*_KNEE_PITCH_.*": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ANKLE_PITCH_.*",
                ".*_ANKLE_ROLL_.*",
            ],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[".*_WAIST_YAW"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_SHOULDER_PITCH_.*",
                ".*_SHOULDER_ROLL_.*",
                ".*_SHOULDER_YAW_.*",
                ".*_ELBOW_PITCH_.*",
                ".*_ELBOW_YAW_.*",
                ".*_HEAD_YAW",
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

from isaaclab.utils import configclass

from whole_body_tracking.robots.pm01_edu import PM01_EDU_ACTION_SCALE, PM01_EDU_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class PM01EduFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PM01_EDU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = PM01_EDU_ACTION_SCALE
        self.commands.motion.anchor_body_name = "LINK_TORSO_YAW"
        self.commands.motion.body_names = [
            "LINK_HIP_YAW_L",       # pelvis root (left leg)
            "LINK_HIP_ROLL_L",      # left hip
            "LINK_KNEE_PITCH_L",    # left knee
            "LINK_ANKLE_ROLL_L",    # left ankle
            "LINK_HIP_ROLL_R",      # right hip
            "LINK_KNEE_PITCH_R",    # right knee
            "LINK_ANKLE_ROLL_R",    # right ankle
            "LINK_TORSO_YAW",       # torso anchor
            "LINK_SHOULDER_ROLL_L", # left shoulder
            "LINK_ELBOW_YAW_L",     # left elbow end-effector
            "LINK_SHOULDER_ROLL_R", # right shoulder
            "LINK_ELBOW_YAW_R",     # right elbow end-effector
        ]
        # Override base_com to use LINK_TORSO_YAW instead of torso_link
        self.events.base_com.params["asset_cfg"].body_names = "LINK_TORSO_YAW"

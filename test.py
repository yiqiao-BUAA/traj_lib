from traj_lib.utils.logger import get_logger
log = get_logger("traj_lib.test")   # 手动指定想要的名字
print(log)   # 应显示 <Logger traj_lib.test (DEBUG)>
log.warning("via get_logger")

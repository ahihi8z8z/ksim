import logging
from sim.faas import FunctionReplica


logger = logging.getLogger(__name__)

class KFunctionReplica(FunctionReplica):
    locked: bool # thên lock tránh nhiều request cùng sửa 1 replica

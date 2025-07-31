import logging
from sim.faas import FunctionReplica

class KFunctionReplica(FunctionReplica):
    locked: bool # thên lock tránh nhiều request cùng sửa 1 replica
    last_invocation: float # thời điểm request cuối cùng được xử lý
    uuid: str # uuid của replica

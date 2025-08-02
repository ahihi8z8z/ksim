import enum

class AppState(enum.IntEnum):
    NULL = 0
    CONCEIVED = 1
    STARTING = 2
    SUSPENDING = 3
    UNLOADED_MODEL = 4
    LOADING_MODEL =  5
    UNLOADING_MODEL = 6
    LOADED_MODEL = 7
    ACTIVING = 8
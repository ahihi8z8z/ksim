from sim.logging import RuntimeLogger, Record

import logging

logger = logging.getLogger(__name__)

class KRuntimeLogger(RuntimeLogger):
    def __init__(self, clock=None) -> None:
        super().__init__(clock)

    def _store_record(self, record: Record):
        field_str = ", ".join(f"{k}={v}" for k, v in record.fields.items())
        tag_str = ", ".join(f"{k}={v}" for k, v in record.tags.items())
        log_line = f"[METRIC] {record.measurement} at {record.time} | {field_str}"
        if tag_str:
            log_line += f" | {tag_str}"
        logging.debug(log_line)




import sys
import traceback


def _extract_error_details(error: Exception, sys_module) -> str:
    _, _, exc_tb = sys_module.exc_info()

    if exc_tb is None:
        return str(error)

    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    return (
        f"\n"
        f"  File    : {file_name}\n"
        f"  Line    : {line_number}\n"
        f"  Error   : {error_message}\n"
        f"  Traceback:\n"
        f"{''.join(traceback.format_tb(exc_tb))}"
    )


class ChurnModelException(Exception):
    def __init__(self, error: Exception, sys_module):
        self.error_message = _extract_error_details(error, sys_module)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        return f"ChurnModelException({self.error_message})"
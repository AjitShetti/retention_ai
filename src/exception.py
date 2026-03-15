from __future__ import annotations


def error_message_detail(error: Exception, error_detail) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_number = exc_tb.tb_lineno if exc_tb else "unknown"
    return (
        "Error occurred in python script "
        f"name [{file_name}] line number [{line_number}] error message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message

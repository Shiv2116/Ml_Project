import sys 

def error_message_detail(error,error_details:sys):
    _,_,exe_tb = error_details.exc_info() # gives information about the exception
    file_name = exe_tb.tb_frame.f_code.co_filename
    error_message ="Error occured in python scrip name [{0}] line number [{1}] with error message [{2}]".format(file_name,exe_tb.tb_lineno,str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = f"Error occurred in {file_name} at line {exc_tb.tb_lineno}: {str(error_message)}"
        return error_message

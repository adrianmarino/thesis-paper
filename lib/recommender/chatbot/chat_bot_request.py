class ChatBotRequest:
    def __init__(self, request_prompt, first=True):
        self.__first = first
        self.__request_prompt = request_prompt

    def __get_char_prompt(self):
        if self.__first:
            return f"""
{'-' * len(self.__request_prompt)}
{self.__request_prompt}
{'-' * len(self.__request_prompt)}
(\\bye or enter to exit)\n
"""
        else:
            return ""

    @property
    def content(self):
        request = input(self.__get_char_prompt()).strip()
        return None if request == "\\bye" or len(request) == 0 else request

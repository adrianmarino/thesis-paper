class ChatBotRequestFactory:
    def __init__(self, request_prompt, user_profile, history):
        self.__request_prompt = request_prompt
        self.__user_profile   = user_profile
        self.__history        = history

    def create(self):
        return ChatBotRequest(
            self.__request_prompt,
            self.__user_profile,
            len(self.__history) == 0
        )


class ChatBotRequest:
    def __init__(self, request_prompt, user_profile, first=True):
        self.__first = first
        self.__request_prompt = request_prompt
        self.__user_profile   = user_profile

    def __get_char_prompt(self):
        if self.__first:
            msg = self.__request_prompt.replace('{user_name}', self.__user_profile.name)

            return f"""
{'-' * len(msg)}
{msg}
{'-' * len(msg)}
(\\bye or enter to exit)\n
"""
        else:
            return ""

    @property
    def content(self):
        request = input(self.__get_char_prompt()).strip()
        return None if request == "\\bye" or len(request) == 0 else request

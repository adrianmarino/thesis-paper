PROMPT_WITH_CANDIDATES = """
{user_profile}

{user_history}

{candidates}

Select {limit} candidate movies based on user profile and seen movies.
Return a list of candidate movies with next regex pattern:

Number. Title (release year): Description.
"""


PROMPT_WITHOUT_CANDIDATES = """
{user_profile}

{user_history}

{candidates}

Select {limit} candidate movies based on user profile and seen movies.
Return a list of candidate movies with the specific next format:

* Title (release year): Description.
"""
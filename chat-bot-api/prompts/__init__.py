PROMPT_LOW_INTERACTIONS = """
{user_profile}

{user_history}

{candidates}

Select {limit} candidate movies based on user profile and seen movies.
Return a list of candidate movies with next regex pattern:

Number. Title (release year): Description.
"""


PROMPT_REQUIRED_INTERACTIONS = """
{user_profile}

{user_history}

{candidates}

Select {limit} movies from candidate movies list based on user profile and seen movies.
Return a list of candidate movies with the specific next format:

Number. Title (release year): Description.
"""
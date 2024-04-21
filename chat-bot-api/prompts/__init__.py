PROMPT_LOW_INTERACTIONS = """
{user_profile}

{user_history}

{candidates}

Recommend {limit} movies from candidate movies list, based on user profile and seen movies.
Return only one list of recommendation with next format:

Number. Title (release year): Synopsis.
"""


PROMPT_REQUIRED_INTERACTIONS = """
{user_history}

{candidates}

Recommend {limit} movies from candidate movies list, excluding seen movies and does
not add movies that does not exists in the candidates list.

Return only one list of recommendation with next format:

Number. Title (release year): Synopsis.
"""


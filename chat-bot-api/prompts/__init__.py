PROMPT_LOW_INTERACTIONS = """
{user_profile}

{user_history}

{request}

{candidates}

Recommend {limit} movies from candidate movies list, based on 'question', 'user profile' and 'seen movies'.
Return only a list of recommendations with next format:

Number. Title (release year): Synopsis.

response only has specified content and must have all specified items.
"""


PROMPT_REQUIRED_INTERACTIONS = """
{user_history}

{request}

{candidates}

Recommend {limit} movies from candidate movies list, based on 'question' and 'excluding seen movies' and does
not add movies that does not exists in the candidates list.

Return only a list of recommendations with next format:

Number. Title (release year): Synopsis.

response only has specified content and must have all specified items.
"""
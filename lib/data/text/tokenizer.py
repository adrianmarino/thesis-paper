from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


STOP_WORDS = """
i
me
my
myself
we
our
ours
ourselves
you
your
yours
yourself
yourselves
he
him
his
himself
she
her
hers
herself
it
its
itself
they
them
their
theirs
themselves
what
which
who
whom
this
that
these
those
am
is
are
was
were
be
been
being
have
has
had
having
do
does
did
doing
a
an
the
and
but
if
or
because
as
until
while
of
at
by
for
with
about
against
between
into
through
during
before
after
above
below
to
from
up
down
in
out
on
off
over
under
again
further
then
once
here
there
when
where
why
how
all
any
both
each
few
more
most
other
some
such
no
nor
not
only
own
same
so
than
too
very
s
t
can
will
just
don
should
now
"""

class TokenizerService:
    def __init__(self, nlp = English(), extra_stop_words=STOP_WORDS):
        self._tokenizer = nlp.tokenizer
        self._stopwords = nlp.Defaults.stop_words
        [self._stopwords.add(word) for word in STOP_WORDS.split('\n')]

    def __call__(self, text):
        return [token.text for token in self._tokenizer(text.lower()) if not self._is_stopword(token)]

    def _is_stopword(self, token):
        return token.is_stop or token.is_punct or token.like_num or token.text in self._stopwords
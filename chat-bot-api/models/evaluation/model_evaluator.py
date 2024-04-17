from IPython.display import clear_output
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import pytorch_common.util as pu
import util as ut
import client


def sample(df, size):
    return df.sample(frac=1).head(size)


def is_in(df1, left_col, df2, right_col):
    return df1[df1[left_col].isin(df2[right_col])]


def is_remaining(df, api_client, profile):
    user_voted_item_ids = [
        int(i.item_id) for i in api_client.interactions_by_user(profile.email)
    ]
    return df[~df["item_id"].isin(user_voted_item_ids)]


def n_items(df):
    return len(df["item_id"].unique())


class RecQueryBuilder:
    def __init__(self, settings):
        self.settings = settings

    def user(self, value):
        self.user_id = value
        return self

    def prompt(self, value):
        self.prompt = value
        return self

    def build(self):
        return {
            "message": {"author": self.user_id, "content": self.prompt},
            "settings": self.settings,
        }


class ModelEvaluator:
    def __init__(
        self,
        evaluation_state,
        interactions_test_set,
        items,
        path,
        api_client,
        verbose=True,
    ):
        self.evaluation_state = evaluation_state
        self.interactions_by_user_id_df = interactions_test_set.sort_values(
            by=["timestamp"]
        ).groupby(["user_id"])
        self.items = items
        self.path = path
        self.api_client = api_client
        self.api_item_ids = [
            int(item["id"]) for item in self.api_client.items(all=True)
        ]
        self.verbose = verbose

    def run(self):
        user_ids = self.interactions_by_user_id_df.groups.keys()

        times = 1
        self.api_client.verbose_off
        for user_id in user_ids:
            profile = self.evaluation_state.find_profile_by_user_id(user_id)

            log_prefix = f"{times}/{len(user_ids)} - {profile.email} - "
            if self.verbose:
                logging.info(f"{log_prefix}Begin user evaluation.")

            interactions_df = self.interactions_by_user_id_df.get_group((user_id,))

            patience = 0
            while True:
                remaining_interactions_df = interactions_df.pipe(
                    is_remaining, self.api_client, profile
                )

                if (
                    self.evaluation_state.was_evaluated(user_id)
                    and len(remaining_interactions_df)
                    < self.evaluation_state.recommendation_size
                ):
                    if self.verbose:
                        logging.info(
                            f"{log_prefix}End evaluation. Less than {self.evaluation_state.recommendation_size} relevant items."
                        )
                    break

                remaining_interactions_sample_df = remaining_interactions_df.pipe(
                    sample, self.evaluation_state.recommendation_size
                )
                if len(remaining_interactions_sample_df) <= 1:
                    break

                remaining_items_sample = self.items.pipe(
                    is_in, "movie_id", remaining_interactions_sample_df, "item_id"
                )

                if self.verbose:
                    logging.info(
                        f"{log_prefix}Sample: {len(remaining_items_sample)} - Remaining: {remaining_interactions_df.pipe(n_items)}/{len(interactions_df)}."
                    )

                try:
                    result, prompt = self.make_request(
                        self.api_client,
                        self.evaluation_state.hyper_params,
                        log_prefix,
                        profile,
                        remaining_items_sample,
                    )
                except client.NotFoundException as err:
                    if self.verbose:
                        logging.info(
                            f"{log_prefix}End user evaluation. Retrain CF models to continue evaluation."
                        )
                    break

                except Exception as err:
                    if self.verbose:
                        logging.error(f"{log_prefix}Api client error. Detail: {err}")
                    continue
                finally:
                    del remaining_items_sample

                rating_by_remaining_item_id = remaining_interactions_sample_df.pipe(
                    ut.to_dict, "item_id", "rating"
                )
                del remaining_interactions_sample_df

                result_relevant_items = list(
                    filter(
                        lambda x: int(x.id) in rating_by_remaining_item_id.keys(),
                        result.items,
                    )
                )

                if len(result_relevant_items) <= 1:
                    max_patience = self.evaluation_state.get_max_patience(
                        len(remaining_interactions_df)
                    )

                    if patience >= max_patience:
                        if self.verbose:
                            logging.warning(
                                f"{log_prefix}End user evaluation. {max_patience} max retries reached."
                            )
                        break

                    patience += 1
                    if self.verbose:
                        logging.info(
                            f"{log_prefix}Not found relevant items into list of {len(result.items)} recommendations - Patience: {patience}/{max_patience}."
                        )
                    continue
                else:
                    patience = 0

                del remaining_interactions_df

                votes = []
                for item in result.items:
                    if int(item.id) in rating_by_remaining_item_id:
                        rating = rating_by_remaining_item_id[int(item.id)]
                        item.vote(rating)
                        votes.append(f"{int(item.id)}->{rating}")
                if self.verbose:
                    logging.info(
                        f'{log_prefix}Found {len(result_relevant_items)} relevant items from a list of {len(result.items)} - Votes: {", ".join(votes)}.'
                    )

                self.evaluation_state.save_session(
                    user_id,
                    session={
                        "recommended_items": [item.id for item in result.items],
                        "recommended_item_ratings": {
                            int(item.id): item.rating for item in result.items
                        },
                        "relevant_item_ratings": rating_by_remaining_item_id,
                    },
                )

                del result
                del rating_by_remaining_item_id

            self.evaluation_state.save(self.path)
            logging.info(f"State saved into {self.path}")

            if times % self.evaluation_state.plot_interval == 0:
                self.plot()

            times += 1

        self.api_client.verbose_on

    def plot(self):
        clear_output(wait=True)
        self.evaluation_state.plot_metrics(self.api_item_ids)

    def make_request(
        self, api_client, hyper_params, log_prefix, profile, remaining_items_sample
    ):
        titles = f"\n".join(
            [f"- {e}" for e in remaining_items_sample["movie_title"].values.tolist()]
        )
        prompt = f"I want to see:\n{titles}"

        sw = pu.Stopwatch()
        response = api_client.recommendations(
            RecQueryBuilder(hyper_params).user(profile.email).prompt(prompt).build()
        )
        if logging.DEBUG >= logging.root.level:
            logging.debug(f"{log_prefix}Response time: {sw.to_str()} - Promp: {prompt}")
        else:
            logging.info(f"{log_prefix}Response time: {sw.to_str()}")

        return response, prompt

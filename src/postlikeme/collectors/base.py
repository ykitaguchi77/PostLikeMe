"""Abstract base class for tweet collectors.

Every concrete collector (Tweepy, twscrape, archive import) must implement
this interface so the rest of the system stays agnostic about the data source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from postlikeme.models.raw_tweet import RawTweet, UserProfile


class BaseCollector(ABC):
    """Asynchronous interface for collecting tweets and user data from X/Twitter.

    Implementations must handle their own authentication, rate-limiting, and
    pagination internally.  Callers interact only with these three methods.
    """

    @abstractmethod
    async def collect_user_tweets(
        self,
        username: str,
        count: int = 500,
        include_replies: bool = True,
    ) -> list[RawTweet]:
        """Collect up to *count* recent tweets for *username*.

        Parameters
        ----------
        username:
            Twitter handle (without the ``@`` prefix).
        count:
            Maximum number of tweets to retrieve.  The actual number may be
            lower if the account has fewer tweets or rate limits are hit.
        include_replies:
            If ``True`` (default), include tweets that are replies to other
            users.  If ``False``, only standalone tweets and quote tweets are
            returned.

        Returns
        -------
        list[RawTweet]
            Collected tweets in reverse-chronological order (newest first).
        """
        ...

    @abstractmethod
    async def collect_tweet_by_id(self, tweet_id: str) -> RawTweet:
        """Fetch a single tweet by its ID.

        Parameters
        ----------
        tweet_id:
            The numeric tweet identifier as a string.

        Returns
        -------
        RawTweet
            The requested tweet.

        Raises
        ------
        LookupError
            If the tweet does not exist or is not accessible.
        """
        ...

    @abstractmethod
    async def get_user_profile(self, username: str) -> UserProfile:
        """Fetch public profile information for *username*.

        Parameters
        ----------
        username:
            Twitter handle (without the ``@`` prefix).

        Returns
        -------
        UserProfile
            Basic public profile data.

        Raises
        ------
        LookupError
            If the user does not exist or is suspended.
        """
        ...

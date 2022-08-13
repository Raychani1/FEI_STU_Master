# SOURCE:
# https://medium.com/@maxtingle/getting-started-with-spotifys-api-spotipy-197c3dc6353b
# https://spotipy.readthedocs.io/en/2.16.1/

import spotipy
from typing import List
from spotipy.oauth2 import SpotifyOAuth


class APICaller:

    def __init__(self) -> None:
        """Initialize the APICaller Class."""

        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(scope="user-library-read")
        )

    def get_market(self, track: str) -> List[str]:
        """Get Market for a given Track.

        Args:
            track (str): Spotify Track ID

        Returns:
            List[str]: List of ISO 3166-1 alpha-2 Country Names

        """

        return self.sp.track(track_id=track)['available_markets']

from enum import Enum


class CorrectionType(Enum):
    LCSUBSTR: str = 'Longest Common Substring'
    LCSUBSEQ: str = 'Longest Common Subsequence'
    EDITDIST: str = 'Edit Distance'

    def __str__(self):
        return self.value

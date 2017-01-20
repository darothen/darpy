#!/usr/bin/env python

from marc_analysis.analysis import _get_masks as get_masks
from marc_analysis.utilities import append_history

if __name__ == "__main__":
    masks = get_masks()
    print("Initial history...")
    print(masks.history)

    print("\nPost-appending history...")
    masks = append_history(masks, extra_info="This is a comment")
    print(masks.history)
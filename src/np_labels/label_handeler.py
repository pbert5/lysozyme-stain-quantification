from __future__ import annotations


class LabelHandeler:
    def __init__(self, labels: list[int]):
        """
        Initialize the LabelHandeler with a list of labels.

        Parameters:
            labels (list[int]): A list of integer labels.
        """
        self.labels = labels
    def merge_labels(self) -> LabelHandeler:
        ... # Placeholder for the actual implementation of merging labels
        return self # This method should contain the logic to merge labels based on specific criteria., should be able to chain together
    def flood_fill(self, blobs, positive_mask) -> LabelHandeler:
        """
        Perform flood fill on the labels.

        Returns:
            LabelHandeler: The instance with flood-filled labels.
        """
        ...
        return self
    
    
            

    
    
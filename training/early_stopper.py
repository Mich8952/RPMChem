class EarlyStopper:
    """Early stopping based on validation loss.

    Patience counts evaluation steps, not epochs.
    E.g. patience=5 means 5 * eval_every training steps.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_curr_model = None

    def __call__(self, val_loss, curr_model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_curr_model = curr_model
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True, self.best_curr_model
        return False, None
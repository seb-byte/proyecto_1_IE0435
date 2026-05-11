import numpy as np


class ThresholdedSVM:
    def __init__(self, model, threshold=0.0):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        scores = self.model.decision_function(X)
        return (scores >= self.threshold).astype(int)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict_proba(self, X):
        scores = self.model.decision_function(X)
        proba_pos = 1 / (1 + np.exp(-scores))
        proba_neg = 1 - proba_pos
        return np.vstack([proba_neg, proba_pos]).T


class ThresholdedRF:
    def __init__(self, pipeline, threshold=0.5):
        self.pipeline  = pipeline
        self.threshold = threshold

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


class SVMRFHybrid:
    """SVM makes the primary prediction; RF (trained with SVM-error samples upweighted)
    provides a correction signal.  Final probability = alpha*SVM + (1-alpha)*RF.
    Both alpha and threshold are tuned on OOF data during training.

    When stacking=True the RF was trained on [X | svm_prob], so predict time
    appends the SVM probability as an extra column before calling the RF."""

    def __init__(self, svm_pipeline, rf_pipeline, blend_alpha=0.5, threshold=0.5,
                 stacking=False):
        self.svm_pipeline = svm_pipeline
        self.rf_pipeline  = rf_pipeline
        self.blend_alpha  = blend_alpha   # weight of SVM in the blend (0-1)
        self.threshold    = threshold
        self.stacking     = stacking

    def _blend(self, X):
        svm_prob = self.svm_pipeline.predict_proba(X)[:, 1]
        X_rf     = np.hstack([X, svm_prob.reshape(-1, 1)]) if self.stacking else X
        rf_prob  = self.rf_pipeline.predict_proba(X_rf)[:, 1]
        return self.blend_alpha * svm_prob + (1.0 - self.blend_alpha) * rf_prob

    def predict(self, X):
        return (self._blend(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        combined = self._blend(X)
        return np.column_stack([1.0 - combined, combined])


class RFSVMHybrid:
    """RF makes the primary prediction; SVM (trained with RF-error samples upweighted)
    provides a correction signal.  Final probability = alpha*RF + (1-alpha)*SVM.
    Both alpha and threshold are tuned on OOF data during training.

    When stacking=True the SVM was trained on [X | rf_prob], so predict time
    appends the RF probability as an extra column before calling the SVM."""

    def __init__(self, rf_pipeline, svm_pipeline, blend_alpha=0.5, threshold=0.5,
                 stacking=False):
        self.rf_pipeline  = rf_pipeline
        self.svm_pipeline = svm_pipeline
        self.blend_alpha  = blend_alpha   # weight of RF in the blend (0-1)
        self.threshold    = threshold
        self.stacking     = stacking

    def _blend(self, X):
        rf_prob = self.rf_pipeline.predict_proba(X)[:, 1]
        X_svm   = np.hstack([X, rf_prob.reshape(-1, 1)]) if self.stacking else X
        svm_prob = self.svm_pipeline.predict_proba(X_svm)[:, 1]
        return self.blend_alpha * rf_prob + (1.0 - self.blend_alpha) * svm_prob

    def predict(self, X):
        return (self._blend(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        combined = self._blend(X)
        return np.column_stack([1.0 - combined, combined])

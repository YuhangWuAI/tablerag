# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .source_features import SourceFeatures


class DataFeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def ExtractTableFeatures(ct, language):
        source_features = DataFeatureExtractor.ConstructSourceFeaturesFromTable(
            ct, language
        )
        return source_features

    @staticmethod
    def ConstructSourceFeaturesFromTable(ct, language):
        sf = SourceFeatures(ct)
        return sf

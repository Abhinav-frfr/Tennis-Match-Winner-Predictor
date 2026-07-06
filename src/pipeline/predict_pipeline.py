import os
import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



import pandas as pd
import sys

class CustomData:
    def __init__(
        self,
        court: str,
        surface: str,
        round: str,
        best_of: int,
        h2h_winpct: float,
        recentwinpct_1: float,
        recentwinpct_2: float,
        season: str,
        surface_winpct_1: float,
        surface_winpct_2: float,
        higher_rank_player: int,
        abs_rank_diff: int,
    ):

        self.court = court
        self.surface = surface
        self.round = round
        self.best_of = best_of
        self.h2h_winpct = h2h_winpct
        self.recentwinpct_1 = recentwinpct_1
        self.recentwinpct_2 = recentwinpct_2
        self.season = season
        self.surface_winpct_1 = surface_winpct_1
        self.surface_winpct_2 = surface_winpct_2
        self.higher_rank_player = higher_rank_player
        self.abs_rank_diff = abs_rank_diff

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Court": [self.court],
                "Surface": [self.surface],
                "Round": [self.round],
                "Best of": [self.best_of],
                "Player1_H2H_WinPct": [self.h2h_winpct],
                "RecentWinPct_1": [self.recentwinpct_1],
                "RecentWinPct_2": [self.recentwinpct_2],
                "Season": [self.season],
                "Surface_WinPct_1": [self.surface_winpct_1],
                "Surface_WinPct_2": [self.surface_winpct_2],
                "Higher_Rank_Player": [self.higher_rank_player],
                "Abs_Rank_Diff": [self.abs_rank_diff],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


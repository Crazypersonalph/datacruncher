from typing import List, Optional, cast
from statistics import median
import dotenv
import tbapy
import os
import pandas as pd


import model

dotenv.load_dotenv()

tba = tbapy.TBA(os.getenv("X-TBA-AUTH-KEY"))

df = pd.DataFrame(columns=["Index", "Team", "Year", "Max Alliance Points", "Rank", "Max Points scored % Qualis", "Max Points scored % Playoffs",
                           "Median Points scored % Qualis", "Median Points scored % Playoffs",
                           "OPR", "DPR", "CCWM", "Win Rate"]).set_index("Index")

max_alliance_points: dict[int, int] = {2023: 217, 2024: 192, 2025: 301}
finals: list[model.CompLevel] = [model.CompLevel.f, model.CompLevel.sf, model.CompLevel.ef, model.CompLevel.qf]


def get_team_score(team: str, event: str, comp_level: list[model.CompLevel]) -> list[int]:
    matches: Optional[List[model.Match]] = cast(Optional[List[model.Match]], tba.team_matches(team, event=event))
    scores: list[int] = []
    
    for match in matches or []:
        match_real = model.Match.model_validate(match)
        for alliance in match_real.alliances.sides():
            if alliance.team_keys.count(team) > 0 and any(x == match_real.comp_level for x in comp_level):
                scores.append(alliance.score)
    return scores

def get_team_rank(team: str, event: str) -> Optional[dict[str, int]]:
    team_event_status: Optional[dict[str, Optional[model.TeamEventStatus]]] = cast(Optional[dict[str, Optional[model.TeamEventStatus]]], tba.event_teams(event, status=True))
    current_event_status = team_event_status.get(team) if team_event_status else None
    current_event_status = model.TeamEventStatus.model_validate(current_event_status) if current_event_status else None
    wins = None
    losses = None
    ties = None
    rank = None
    matches_played = None
    if current_event_status:
        if current_event_status.qual:
            if current_event_status.qual.ranking:
                rank = current_event_status.qual.ranking.rank
                if current_event_status.qual.ranking.record:
                    wins = current_event_status.qual.ranking.record.wins
                    losses = current_event_status.qual.ranking.record.losses
                    ties = current_event_status.qual.ranking.record.ties
                    matches_played = wins + losses + ties

        if current_event_status.playoff:
            if current_event_status.playoff.record:
                if wins is not None and losses is not None and ties is not None:
                    wins += current_event_status.playoff.record.wins
                    losses += current_event_status.playoff.record.losses
                    ties += current_event_status.playoff.record.ties
                    matches_played = wins + losses + ties
            
    return {"wins": wins, "losses": losses, "ties": ties, "rank": rank, "matches_played": matches_played} if wins is not None and losses is not None and ties is not None and rank is not None and matches_played is not None else None

#def print_team_scores(team: str, event: str, comp_level: list[model.CompLevel]) -> None:
#    scores = get_team_score(team, event, comp_level)
#    print(f"Scores for team {team} at event {event} for levels {[x.value for x in comp_level]}:")
#    print("Scores:", scores)
#    if scores:
#        print("Max score:", max(scores))
#        print("Median score:", median(sorted(scores)))
#    else:
#        print("No scores available.")
#    print()

def team_db(team: str) -> None:
    for i in range(3):
        year = 2023 + i
        event_code = f"{year}auwarp"
        row = [team, year, max_alliance_points[year], None, None, None, None, None, None, None, None, None]

        
        rank = get_team_rank(team, event_code)
        if rank is not None:
            if rank.get("rank") is not None: # Actual Rank
                row[3] = rank["rank"]
            if rank.get("matches_played") is not None and rank.get("wins") is not None: # Win/Loss Percentage
                total_matches = rank["matches_played"]
                if total_matches > 0:
                    win_rate = (rank["wins"] / total_matches) * 100
                    row[11] = win_rate
        
        
        # Quals
        quals_scores = get_team_score(team, event_code, [model.CompLevel.qm])
        if quals_scores:
            row[4] = (max(quals_scores) / max_alliance_points[year]) * 100
            row[6] = (median(sorted(quals_scores)) / max_alliance_points[year]) * 100

        # Playoffs
        playoffs_scores = get_team_score(team, event_code, finals)
        if playoffs_scores:
            row[5] = (max(playoffs_scores) / max_alliance_points[year]) * 100
            row[7] = (median(sorted(playoffs_scores)) / max_alliance_points[year]) * 100

        # OPR, DPR, CCWM, and Win Rate
        event_stats: Optional[model.EventOPRs] = cast(Optional[model.EventOPRs], tba.event_oprs(event_code))
        if event_stats:
            oprs = event_stats.oprs
            if oprs and team in oprs:
                row[8] = oprs[team]
            dprs = event_stats.dprs
            if dprs and team in dprs:
                row[9] = dprs[team]
            ccwms = event_stats.ccwms
            if ccwms and team in ccwms:
                row[10] = ccwms[team]

        df.loc[f"{team}_{year}"] = row
team_db("frc4788")
team_db("frc10342")
team_db("frc7113")
team_db("frc9975")
df.to_csv("warp_data.csv")

# there is technically an edge case in that it looks 
# over the past 3 years for all teams 
# (including off-season, whose numbers may change)
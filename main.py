from typing import List, Optional, cast
from statistics import median
import dotenv
import tbapy
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import model

dotenv.load_dotenv()

tba = tbapy.TBA(os.getenv("X-TBA-AUTH-KEY"))

parser = argparse.ArgumentParser(
                    prog="Alphons's small FIRST data cruncher",
                    description='Takes in data from The Blue Alliance API and crunches some numbers.',
                    epilog='Enjoy')
parser.add_argument('-t', '--teams', type=str, nargs='+', help='List of team keys to analyze (e.g., frc4788 frc10342)', required=True)
parser.add_argument('-e', '--eventcode', type=str, help='Event code suffix to analyze (e.g., auwarp for AU Warp)', required=True)
parser.add_argument('-o', '--output', type=str, help='Output CSV file name', default='warp_data.csv')
parser.add_argument('--year-range', type=int, nargs=2, help='Year range to analyze (e.g., 2023 2025) inclusive', default=[2023, 2025])

args = parser.parse_args()

df = pd.DataFrame(columns=["Index", "Team", "Year", "Max Alliance Points", "Rank", "Max Points scored % Qualis", "Max Points scored % Playoffs",
                           "Median Points scored % Qualis", "Median Points scored % Playoffs",
                           "OPR", "DPR", "CCWM", "Win Rate"]).set_index("Index")

max_alliance_points: dict[int, int] = {2023: 217, 2024: 192, 2025: 301}

finals: list[model.CompLevel] = [model.CompLevel.f, model.CompLevel.sf, model.CompLevel.ef, model.CompLevel.qf]

def get_year_max(year: int) -> int:
    leaderboardInsight: list[model.LeaderboardInsight] = cast(list[model.LeaderboardInsight], tba.insights_leaderboards(year))
    max_value: int = 0
    for insight in leaderboardInsight:
        real_insight = model.LeaderboardInsight.model_validate(insight)
        if real_insight.name == "typed_leaderboard_highest_match_clean_score":
            for entry in real_insight.data.rankings:
                if entry.value > max_value:
                    max_value = int(entry.value)
    return max_value

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

def team_db(team: str, event_specific_code: str) -> None:
    for i in range(args.year_range[1] - args.year_range[0] + 1):
        year = args.year_range[0] + i

        event_code = f"{year}{event_specific_code}"
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

for team in args.teams:
    team_db(team, args.eventcode)
df.to_csv(args.output)

# there is technically an edge case in that it looks 
# over the past 3 years for all teams 
# (including off-season, whose numbers may change)

# df columns: ['team', 'x', 'y']
teams = df['Team'].unique()
palette = plt.get_cmap('tab20', len(teams))
colors = dict(zip(teams, palette(np.linspace(0, 1, len(teams)))))

for team in teams:
    g = df[df['Team'] == team]
    
    # Max Points Qualis (solid line)
    g_qualis_max = g.dropna(subset=['Max Points scored % Qualis'])
    if len(g_qualis_max) == 1:
        plt.scatter(g_qualis_max['Year'], g_qualis_max['Max Points scored % Qualis'], 
                   color=colors[team], s=100, marker='o', label=f'{team} - Max Qualis')
    elif len(g_qualis_max) > 1:
        plt.plot(g_qualis_max['Year'], g_qualis_max['Max Points scored % Qualis'], 
                color=colors[team], linestyle='-', marker='o', label=f'{team} - Max Qualis')
    
    # Median Points Qualis (dotted line)
    g_qualis_med = g.dropna(subset=['Median Points scored % Qualis'])
    if len(g_qualis_med) == 1:
        plt.scatter(g_qualis_med['Year'], g_qualis_med['Median Points scored % Qualis'], 
                   color=colors[team], s=100, marker='s')
    elif len(g_qualis_med) > 1:
        plt.plot(g_qualis_med['Year'], g_qualis_med['Median Points scored % Qualis'], 
                color=colors[team], linestyle=':', marker='s', label=f'{team} - Median Qualis')
    
    # Max Points Finals (dashed line)
    g_finals_max = g.dropna(subset=['Max Points scored % Playoffs'])
    if len(g_finals_max) == 1:
        plt.scatter(g_finals_max['Year'], g_finals_max['Max Points scored % Playoffs'], 
                   color=colors[team], s=100, marker='^')
    elif len(g_finals_max) > 1:
        plt.plot(g_finals_max['Year'], g_finals_max['Max Points scored % Playoffs'], 
                color=colors[team], linestyle='--', marker='^', label=f'{team} - Max Finals')
    
    # Median Points Finals (dashdot line)
    g_finals_med = g.dropna(subset=['Median Points scored % Playoffs'])
    if len(g_finals_med) == 1:
        plt.scatter(g_finals_med['Year'], g_finals_med['Median Points scored % Playoffs'], 
                   color=colors[team], s=100, marker='D')
    elif len(g_finals_med) > 1:
        plt.plot(g_finals_med['Year'], g_finals_med['Median Points scored % Playoffs'], 
                color=colors[team], linestyle='-.', marker='D', label=f'{team} - Median Finals')

plt.legend(title='Legend', loc='upper left', fontsize=6)
plt.xlabel('Year')
plt.ylabel('Percentage of Max Points Scored (%)')
plt.title('Team Performance Comparison')
plt.tight_layout()
plt.xticks(range(args.year_range[0], args.year_range[1] + 1))
plt.yticks(range(0, 101, 10))
plt.show()
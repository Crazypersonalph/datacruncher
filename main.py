from typing import List, Optional, cast, Iterable
from statistics import median
import dotenv
import tbapy
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
import model
import logging
import time

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

# Create an empty list of rows (dicts) to avoid concat warnings on empty dataframes
rows: list[dict[str, Optional[float | int | str]]] = []


finals: list[model.CompLevel] = [model.CompLevel.f, model.CompLevel.sf, model.CompLevel.ef, model.CompLevel.qf]

# Simple in-memory caches to avoid repeated API calls
_year_max_cache: dict[int, Optional[int]] = {}
_event_status_cache: dict[str, Optional[dict[str, Optional[model.TeamEventStatus]]]] = {}
_event_oprs_cache: dict[str, Optional[model.EventOPRs]] = {}
_team_matches_cache: dict[tuple[str, str], Optional[List[model.Match]]] = {}  # (team, event) -> matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("datacruncher.log", mode="w", encoding="utf-8")],
)

# TBA API wrapper functions with logging and timing
def tba_insights_leaderboards(year: int):
    logging.info("TBA call start: insights_leaderboards year=%s", year)
    t0 = time.perf_counter()
    result = tba.insights_leaderboards(year)
    logging.info("TBA call end: insights_leaderboards (%.3fs)", time.perf_counter() - t0)
    return result

def tba_event_teams(event: str, *, status: bool = True):
    logging.info("TBA call start: event_teams event=%s status=%s", event, status)
    t0 = time.perf_counter()
    result = tba.event_teams(event, status=status)
    logging.info("TBA call end: event_teams (%.3fs)", time.perf_counter() - t0)
    return result

def tba_event_oprs(event: str):
    logging.info("TBA call start: event_oprs event=%s", event)
    t0 = time.perf_counter()
    result = tba.event_oprs(event)
    logging.info("TBA call end: event_oprs (%.3fs)", time.perf_counter() - t0)
    return result

def tba_team_matches(team: str, *, event: str):
    logging.info("TBA call start: team_matches team=%s event=%s", team, event)
    t0 = time.perf_counter()
    result = tba.team_matches(team, event=event)
    logging.info("TBA call end: team_matches (%.3fs)", time.perf_counter() - t0)
    return result

def get_year_max(year: int) -> Optional[int]:
    if year in _year_max_cache:
        return _year_max_cache[year]

    leaderboardInsight: list[model.LeaderboardInsight] = cast(
        list[model.LeaderboardInsight],
        tba_insights_leaderboards(year),
    )
    max_value: Optional[int] = None
    # TBA API v3 Docs
    for insight in leaderboardInsight:
        real_insight = model.LeaderboardInsight.model_validate(insight)
        if real_insight.name == "typed_leaderboard_highest_match_clean_score":
            for entry in real_insight.data.rankings:
                if max_value is None or entry.value > max_value:
                    max_value = int(entry.value)

    _year_max_cache[year] = max_value
    return max_value

def get_team_score(team: str, event: str, comp_level: list[model.CompLevel]) -> list[int]:
    key = (team, event)
    if key not in _team_matches_cache:
        _team_matches_cache[key] = cast(
            Optional[List[model.Match]],
            tba_team_matches(team, event=event),
        )

    matches = _team_matches_cache[key]
    scores: list[int] = []
    comp_set = set(comp_level)

    for match in matches or []:
        match_real = model.Match.model_validate(match)
        if match_real.comp_level not in comp_set:
            continue
        for alliance in match_real.alliances.sides():
            if team in alliance.team_keys:
                scores.append(alliance.score)
                break

    return scores

def get_team_rank(team: str, event: str) -> Optional[dict[str, int]]:
    if event not in _event_status_cache:
        _event_status_cache[event] = cast(
            Optional[dict[str, Optional[model.TeamEventStatus]]],
            tba_event_teams(event, status=True),
        )

    team_event_status = _event_status_cache[event]
    current_event_status = team_event_status.get(team) if team_event_status else None
    current_event_status = model.TeamEventStatus.model_validate(current_event_status) if current_event_status else None

    wins = None
    losses = None
    ties = None
    rank = None
    matches_played = None

    if current_event_status:
        if current_event_status.qual and current_event_status.qual.ranking:
            rank = current_event_status.qual.ranking.rank
            if current_event_status.qual.ranking.record:
                wins = current_event_status.qual.ranking.record.wins
                losses = current_event_status.qual.ranking.record.losses
                ties = current_event_status.qual.ranking.record.ties
                matches_played = wins + losses + ties

        if current_event_status.playoff and current_event_status.playoff.record:
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
    start_year, end_year = args.year_range
    for year in range(start_year, end_year + 1):
        year_max = get_year_max(year)
        if year_max is None:
            warnings.warn(
                f"No max score data available for year {year}. Using fallback value of 300.",
                UserWarning,
                stacklevel=90,
            )
            year_max = 300

        event_code = f"{year}{event_specific_code}"
        row: dict[str, Optional[float | int | str]] = {
            "Index": f"{team}_{year}",
            "Team": team,
            "Year": year,
            "Max Alliance Points": year_max,
            "Max Alliance Points exists": year_max is not None,
            "Rank": None,
            "Max Points scored % Qualis": None,
            "Max Points scored % Playoffs": None,
            "Median Points scored % Qualis": None,
            "Median Points scored % Playoffs": None,
            "OPR": None,
            "DPR": None,
            "CCWM": None,
            "Win Rate": None,
        }

        
        rank = get_team_rank(team, event_code)
        if rank is not None:
            if rank.get("rank") is not None: # Actual Rank
                row["Rank"] = rank["rank"]
            if rank.get("matches_played") is not None and rank.get("wins") is not None: # Win/Loss Percentage
                total_matches = rank["matches_played"]
                if total_matches > 0:
                    win_rate = (rank["wins"] / total_matches) * 100
                    row["Win Rate"] = win_rate
        
        
        # Quals
        quals_scores = get_team_score(team, event_code, [model.CompLevel.qm])
        if quals_scores:
            row["Max Points scored % Qualis"] = (max(quals_scores) / year_max) * 100
            row["Median Points scored % Qualis"] = (median(sorted(quals_scores)) / year_max) * 100

        # Playoffs
        playoffs_scores = get_team_score(team, event_code, finals)
        if playoffs_scores:
            row["Max Points scored % Playoffs"] = (max(playoffs_scores) / year_max) * 100
            row["Median Points scored % Playoffs"] = (median(sorted(playoffs_scores)) / year_max) * 100

        if event_code not in _event_oprs_cache:
            _event_oprs_cache[event_code] = cast(
                Optional[model.EventOPRs],
                tba_event_oprs(event_code),
            )

        event_stats = _event_oprs_cache[event_code]
        if event_stats:
            if event_stats.oprs and team in event_stats.oprs:
                row["OPR"] = event_stats.oprs[team]
            if event_stats.dprs and team in event_stats.dprs:
                row["DPR"] = event_stats.dprs[team]
            if event_stats.ccwms and team in event_stats.ccwms:
                row["CCWM"] = event_stats.ccwms[team]

        rows.append(row)

for team in args.teams:
    team_db(team, args.eventcode)
df = pd.DataFrame(rows).set_index("Index")
df.to_csv(args.output, na_rep="")

# there is technically an edge case in that it looks 
# over the past 3 years for all teams 
# (including off-season, whose numbers may change)

# df columns: ['team', 'x', 'y']
teams = list(df['Team'].unique())
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


plt.legend(title='Legend', loc='upper left', fontsize=6, borderpad=1)
plt.xlabel('Year')
plt.ylabel('Percentage of Max Points Scored (%)')
plt.title(f'Team Performance Comparison - {args.eventcode.upper()}')
plt.tight_layout()
plt.xticks(range(args.year_range[0], args.year_range[1] + 1))
plt.yticks(range(0, 101, 10))
plt.show()
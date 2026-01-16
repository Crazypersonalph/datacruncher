# Alphons's tiny FRC number cruncher

Please note that there is a tbapy submodule in this repo with updated changes for the newer TBA API, as the PyPI package has not been updated.

There is also a .env with `X-TBA-AUTH-KEY` which is, you guessed it, populated with the TBA API key.

If the program can't find max score data for a year, it will default to 300. This is most notable in 2021 Infinite Recharge due to COVID.

If there is nothing in the graph, you have put an invalid value(s) in SHOW_PLOTS

```
usage: Alphons's small FIRST data cruncher [-h] -t TEAMS [TEAMS ...] -e EVENTCODE [-o OUTPUT]
                                           [--year-range YEAR_RANGE YEAR_RANGE] [-s SHOW_PLOTS [SHOW_PLOTS ...]]

Takes in data from The Blue Alliance API and crunches some numbers.

options:
  -h, --help            show this help message and exit
  -t TEAMS [TEAMS ...], --teams TEAMS [TEAMS ...]
                        List of team keys to analyze (e.g., frc4788 frc10342)
  -e EVENTCODE, --eventcode EVENTCODE
                        Event code suffix to analyze (e.g., auwarp for AU Warp)
  -o OUTPUT, --output OUTPUT
                        Output CSV file name
  --year-range YEAR_RANGE YEAR_RANGE
                        Year range to analyze (e.g., 2023 2025) inclusive
  -s SHOW_PLOTS [SHOW_PLOTS ...], --show-plots SHOW_PLOTS [SHOW_PLOTS ...]
                        Select which plots to show. However, sometimes it doesn't make sense to show different
                        things together. (rank, max_qualis, median_qualis, max_finals, median_finals, opr, dpr,
                        ccwm, win_rate)

Enjoy
```

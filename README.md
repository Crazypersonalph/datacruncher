# Alphons's tiny FRC number cruncher

Please note that I have made some custom changes to tbapy because it hasn't been updated in a while

However the changes are pretty simple and easy to implement, so that has been left as an exercise for the reader

There is also a .env with `X-TBA-AUTH-KEY` which is, you guessed it, populated with the TBA API key.

If the program can't find max score data for a year, it will default to 100. This is most notable in 2021 Infinite Recharge due to COVID.

```
usage: Alphons's small FIRST data cruncher [-h] -t TEAMS [TEAMS ...] -e EVENTCODE [-o OUTPUT] [--year-range YEAR_RANGE YEAR_RANGE]

Takes in data from The Blue Alliance API and crunches some numbers. (makes a cool graph too)

options:
  -h, --help            show this help message and exit
  -t TEAMS [TEAMS ...], --teams TEAMS [TEAMS ...]
                        List of team keys to analyze (e.g., frc4788 frc10342)
  -e EVENTCODE, --eventcode EVENTCODE
                        Event code suffix to analyze (e.g., auwarp for AU Warp) (can be found on TBA website for event)
  -o OUTPUT, --output OUTPUT
                        Output CSV file name
  --year-range YEAR_RANGE YEAR_RANGE
                        Year range to analyze (e.g., 2023 2025) inclusive

Enjoy
```

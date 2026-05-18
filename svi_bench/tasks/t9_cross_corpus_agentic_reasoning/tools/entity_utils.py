import difflib
from typing import List, Set, Dict, Optional

def resolve_entities(input_names: List[str], canonical_set: Set[str], threshold: float = 0.6) -> List[str]:
    """
    Resolves a list of input names to canonical names using fuzzy matching.
    """
    resolved = []
    if not input_names or not canonical_set:
        return resolved

    # Normalize canonical set for easier matching (lowercase)
    # Mapping: lowercase -> original
    canonical_map = {name.lower(): name for name in canonical_set}
    canonical_lower = list(canonical_map.keys())

    for name in input_names:
        name_lower = name.lower()
        # 1. Exact match (case-insensitive)
        if name_lower in canonical_map:
            resolved.append(canonical_map[name_lower])
            continue
        
        # 2. Fuzzy match
        matches = difflib.get_close_matches(name_lower, canonical_lower, n=1, cutoff=threshold)
        if matches:
            resolved.append(canonical_map[matches[0]])
        else:
            # 3. Substring match fallback (if "Clippers" is in "Los Angeles Clippers")
            best_match = None
            for c_lower in canonical_lower:
                if name_lower in c_lower or c_lower in name_lower:
                    best_match = canonical_map[c_lower]
                    break
            if best_match:
                resolved.append(best_match)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(resolved))

# Common aliases mapping
TEAM_ALIASES = {
    "lakers": "Los Angeles Lakers",
    "clippers": "Los Angeles Clippers",
    "warriors": "Golden State Warriors",
    "sixers": "Philadelphia 76ers",
    "mavs": "Dallas Mavericks",
    "cavs": "Cleveland Cavaliers",
    "blazers": "Portland Trail Blazers",
    "wolves": "Minnesota Timberwolves",
    "bucks": "Milwaukee Bucks",
    "bulls": "Chicago Bulls",
    "celtics": "Boston Celtics",
    "knicks": "New York Knicks",
    "nets": "Brooklyn Nets",
    "nuggets": "Denver Nuggets",
    "pacers": "Indiana Pacers",
    "pistons": "Detroit Pistons",
    "raptors": "Toronto Raptors",
    "rockets": "Houston Rockets",
    "spurs": "San Antonio Spurs",
    "suns": "Phoenix Suns",
    "thunder": "Oklahoma City Thunder",
    "jazz": "Utah Jazz",
    "kings": "Sacramento Kings",
    "magic": "Orlando Magic",
    "pelicans": "New Orleans Pelicans",
    "heat": "Miami Heat",
    "hawks": "Atlanta Hawks",
    "hornets": "Charlotte Hornets",
    "grizzlies": "Memphis Grizzlies",
    "wizards": "Washington Wizards"
}

HOCKEY_TEAM_ALIASES = {
    "leafs": "Toronto Maple Leafs",
    "maple leafs": "Toronto Maple Leafs",
    "habs": "Montreal Canadiens",
    "canadiens": "Montreal Canadiens",
    "bruins": "Boston Bruins",
    "rangers": "New York Rangers",
    "islanders": "New York Islanders",
    "penguins": "Pittsburgh Penguins",
    "pens": "Pittsburgh Penguins",
    "flyers": "Philadelphia Flyers",
    "capitals": "Washington Capitals",
    "caps": "Washington Capitals",
    "red wings": "Detroit Red Wings",
    "wings": "Detroit Red Wings",
    "blackhawks": "Chicago Blackhawks",
    "wild": "Minnesota Wild",
    "blues": "St. Louis Blues",
    "predators": "Nashville Predators",
    "preds": "Nashville Predators",
    "stars": "Dallas Stars",
    "avalanche": "Colorado Avalanche",
    "avs": "Colorado Avalanche",
    "jets": "Winnipeg Jets",
    "flames": "Calgary Flames",
    "oilers": "Edmonton Oilers",
    "canucks": "Vancouver Canucks",
    "sharks": "San Jose Sharks",
    "ducks": "Anaheim Ducks",
    "golden knights": "Vegas Golden Knights",
    "knights": "Vegas Golden Knights",
    "kraken": "Seattle Kraken",
    "hurricanes": "Carolina Hurricanes",
    "canes": "Carolina Hurricanes",
    "panthers": "Florida Panthers",
    "lightning": "Tampa Bay Lightning",
    "bolts": "Tampa Bay Lightning",
    "senators": "Ottawa Senators",
    "sens": "Ottawa Senators",
    "sabres": "Buffalo Sabres",
    "devils": "New Jersey Devils",
    "blue jackets": "Columbus Blue Jackets",
    "jackets": "Columbus Blue Jackets",
    "coyotes": "Arizona Coyotes",
}

SOCCER_TEAM_ALIASES = {
    # EPL
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "man city": "Manchester City",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "west ham": "West Ham United",
    "hammers": "West Ham United",
    "villa": "Aston Villa",
    "toffees": "Everton",
    "saints": "Southampton",
    "magpies": "Newcastle United",
    "newcastle": "Newcastle United",
    "forest": "Nottingham Forest",
    "notts forest": "Nottingham Forest",
    "bees": "Brentford",
    "seagulls": "Brighton and Hove Albion",
    "brighton": "Brighton and Hove Albion",
    "palace": "Crystal Palace",
    "clarets": "Burnley",
    "cottagers": "Fulham",
    "hatters": "Luton Town",
    "luton": "Luton Town",
    "cherries": "Bournemouth",
    "bournemouth": "AFC Bournemouth",
    "blades": "Sheffield United",
    "sheffield utd": "Sheffield United",
    # LaLiga
    "barca": "Barcelona",
    "barça": "Barcelona",
    "real": "Real Madrid",
    "atletico": "Atlético Madrid",
    "atleti": "Atlético Madrid",
    "betis": "Real Betis",
    "sociedad": "Real Sociedad",
    "athletic": "Athletic Club",
    "celta": "Celta Vigo",
    "las palmas": "Las Palmas",
    "alaves": "Alavés",
    "cadiz": "Cádiz",
    "almeria": "Almería",
}


_SPORT_ALIASES = {
    "basketball": TEAM_ALIASES,
    "hockey": HOCKEY_TEAM_ALIASES,
    "soccer": SOCCER_TEAM_ALIASES,
}


def normalize_team_name(name: str, sport: str = None) -> str:
    """Pre-resolves common aliases before fuzzy matching.

    Pass ``sport`` so collisions across sports resolve correctly
    (e.g. ``"spurs"`` → "San Antonio Spurs" when sport=basketball, but
    → "Tottenham Hotspur" when sport=soccer). When ``sport`` is not
    given we fall back to the merged dict but with NBA aliases taking
    precedence over EPL ones, matching the historical default for
    ambiguous tokens like "spurs" / "wolves".
    """
    name_clean = name.lower().strip()
    if sport and sport in _SPORT_ALIASES:
        return _SPORT_ALIASES[sport].get(name_clean, name)
    # Caller didn't tell us the sport. Merge with NBA last so its
    # entries override EPL on key collisions.
    merged = {**SOCCER_TEAM_ALIASES, **HOCKEY_TEAM_ALIASES, **TEAM_ALIASES}
    return merged.get(name_clean, name)

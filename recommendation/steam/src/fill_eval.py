import pandas as pd

SRC = "artifacts/s1_v2/ranking_eval_pilot.xlsx"

# (anchor_name, candidate_name) → (relevance, confidence)
SCORES: dict[tuple[str, str], tuple[int, int]] = {
    # Project Zomboid
    ("Project Zomboid", "Invasion: Brain Craving"): (2, 2),
    ("Project Zomboid", "State of Decay: YOSE"): (3, 3),
    ("Project Zomboid", "Dead Age"): (3, 2),
    ("Project Zomboid", "Delivery from the Pain:Survival / 末日方舟:生存"): (2, 2),
    ("Project Zomboid", "Zombie Grinder"): (1, 1),
    ("Project Zomboid", "Infection Rate"): (1, 1),
    ("Project Zomboid", "Rebuild 3: Gangs of Deadsville"): (2, 2),
    ("Project Zomboid", "Exile to Death"): (1, 1),
    ("Project Zomboid", "Flesh Eaters"): (2, 1),
    ("Project Zomboid", "Envy the Dead"): (2, 2),
    ("Project Zomboid", "Judgment: Apocalypse Survival Simulation"): (2, 2),
    ("Project Zomboid", "Girl Amazon Survival"): (1, 1),
    ("Project Zomboid", "Crea"): (1, 1),
    ("Project Zomboid", "Zombie Panic! Source"): (2, 1),

    # Tabletop Simulator
    ("Tabletop Simulator", "Dice 1000 online"): (1, 1),
    ("Tabletop Simulator", "Mutilate-a-Doll 2"): (1, 1),
    ("Tabletop Simulator", "Potemkin"): (1, 1),
    ("Tabletop Simulator", "RollerCoaster Tycoon® Classic"): (2, 2),
    ("Tabletop Simulator", "Universe Sandbox"): (2, 2),
    ("Tabletop Simulator", "Archmage Rises"): (1, 1),
    ("Tabletop Simulator", "Home Improvisation: Furniture Sandbox"): (2, 2),
    ("Tabletop Simulator", "ROD: Revolt Of Defense"): (1, 1),
    ("Tabletop Simulator", "Castaway Paradise - live among the animals"): (1, 1),
    ("Tabletop Simulator", "Brick Rigs"): (2, 2),
    ("Tabletop Simulator", "Rise to Ruins"): (1, 1),
    ("Tabletop Simulator", "Urban Pirate"): (1, 1),
    ("Tabletop Simulator", "Ancient Warfare 3"): (1, 1),
    ("Tabletop Simulator", "Pinball Deluxe: Reloaded"): (1, 1),

    # Company of Heroes 2
    ("Company of Heroes 2", "Victory At Sea"): (1, 1),
    ("Company of Heroes 2", "Sudden Strike 3"): (3, 3),
    ("Company of Heroes 2", "Warhammer 40,000: Dawn of War II - Anniversary Edition (Classic)"): (2, 2),
    ("Company of Heroes 2", "Supremacy: Call of War 1942"): (2, 2),
    ("Company of Heroes 2", "Sudden Strike 2 Gold"): (3, 3),
    ("Company of Heroes 2", "Fog of War"): (1, 1),
    ("Company of Heroes 2", "Sudden Strike Gold"): (3, 3),
    ("Company of Heroes 2", "Commandos 2: Men of Courage"): (2, 2),
    ("Company of Heroes 2", "Blitzkrieg 2 Anthology"): (3, 3),
    ("Company of Heroes 2", "Battle Islands"): (1, 1),
    ("Company of Heroes 2", "Company of Heroes: Opposing Fronts"): (3, 3),

    # Cities: Skylines
    ("Cities: Skylines", "Chris Sawyer's Locomotion™"): (2, 2),
    ("Cities: Skylines", "Hexters"): (1, 1),
    ("Cities: Skylines", "Anno 2070™"): (3, 3),
    ("Cities: Skylines", "CitiesCorp Concept - Build Everything on Your Own"): (2, 2),
    ("Cities: Skylines", "Citystate"): (3, 3),
    ("Cities: Skylines", "CloudCity VR"): (1, 1),
    ("Cities: Skylines", "SimCity™ 4 Deluxe Edition"): (3, 3),
    ("Cities: Skylines", "Jane's Realty"): (1, 1),
    ("Cities: Skylines", "Seven Kingdoms 2 HD"): (2, 2),
    ("Cities: Skylines", "Skytropolis"): (1, 1),

    # Don't Starve
    ("Don't Starve", "Judgment: Apocalypse Survival Simulation"): (2, 2),
    ("Don't Starve", "On My Own"): (2, 2),
    ("Don't Starve", "Landless"): (1, 1),
    ("Don't Starve", "The Isle"): (1, 1),
    ("Don't Starve", "Farlight Explorers"): (1, 1),
    ("Don't Starve", "Lost in Nature"): (2, 2),
    ("Don't Starve", "Whitetail Challenge"): (1, 1),
    ("Don't Starve", "Planetbase"): (1, 1),
    ("Don't Starve", "Forsaken Isle"): (2, 2),
    ("Don't Starve", "Eco"): (2, 2),
    ("Don't Starve", "Exile to Death"): (1, 1),
    ("Don't Starve", "Lost Shipwreck"): (2, 2),
    ("Don't Starve", "Odd Realm"): (1, 1),
    ("Don't Starve", "DwarfCorp"): (1, 1),

    # Raft
    ("Raft", "Landless"): (2, 2),
    ("Raft", "Make Sail"): (2, 2),
    ("Raft", "Escape The Pacific"): (3, 3),
    ("Raft", "Pixel Piracy"): (2, 2),
    ("Raft", "Lost in Nature"): (2, 2),
    ("Raft", "Rule with an Iron Fish - A Pirate Fishing Adventure"): (2, 2),
    ("Raft", "Miner Wars 2081"): (1, 1),
    ("Raft", "Forsaken Isle"): (2, 2),
    ("Raft", "Space Rogue"): (1, 1),
    ("Raft", "Ironclads 2: War of the Pacific"): (0, 0),
    ("Raft", "The Caribbean Sail"): (1, 1),

    # Stardew Valley
    ("Stardew Valley", "World's Dawn"): (3, 3),
    ("Stardew Valley", "Pro Farm Manager"): (2, 2),
    ("Stardew Valley", "My Time at Portia"): (3, 3),
    ("Stardew Valley", "DwarfCorp"): (1, 1),
    ("Stardew Valley", "Staxel"): (2, 2),
    ("Stardew Valley", "홈 저장"): (1, 1),
    ("Stardew Valley", "Planetbase"): (1, 1),
    ("Stardew Valley", "Prosperity"): (2, 2),
    ("Stardew Valley", "Odd Realm"): (1, 1),
    ("Stardew Valley", "Disturbed"): (1, 1),
    ("Stardew Valley", "Castaway Paradise - live among the animals"): (2, 2),
    ("Stardew Valley", "Stonehearth"): (2, 2),

    # Phasmophobia
    ("Phasmophobia", "Unlasting Horror"): (3, 3),
    ("Phasmophobia", "GhostControl Inc."): (2, 2),
    ("Phasmophobia", "Outbreak"): (2, 2),
    ("Phasmophobia", "Xark"): (1, 1),
    ("Phasmophobia", "Killing Floor"): (2, 2),
    ("Phasmophobia", "Spooky Night"): (1, 1),
    ("Phasmophobia", "Death Road to Canada"): (1, 1),
    ("Phasmophobia", "Apparition"): (1, 1),
    ("Phasmophobia", "Zombie Panic! Source"): (2, 2),
    ("Phasmophobia", "Outbreak: The New Nightmare"): (2, 2),
    ("Phasmophobia", "Over My Dead Body (For You)"): (1, 1),
    ("Phasmophobia", "Congo"): (0, 0),
    ("Phasmophobia", "Dead by Daylight"): (3, 3),

    # Left 4 Dead 2
    ("Left 4 Dead 2", "Atom Zombie Smasher"): (1, 1),
    ("Left 4 Dead 2", "Zombie Panic! Source"): (3, 3),
    ("Left 4 Dead 2", "Splatter - Zombiecalypse Now"): (2, 2),
    ("Left 4 Dead 2", "Infection Rate"): (1, 1),
    ("Left 4 Dead 2", "Killing Floor"): (3, 3),
    ("Left 4 Dead 2", "Outbreak"): (2, 2),
    ("Left 4 Dead 2", "Death Road to Canada"): (2, 2),
    ("Left 4 Dead 2", "Zombie Grinder"): (1, 1),
    ("Left 4 Dead 2", "GIBZ"): (1, 1),
    ("Left 4 Dead 2", "One Of The Last"): (1, 1),
    ("Left 4 Dead 2", "Congo"): (0, 0),
    ("Left 4 Dead 2", "Dead Rising® 2"): (3, 3),
    ("Left 4 Dead 2", "Dead by Daylight"): (2, 2),

    # Satisfactory
    ("Satisfactory", "Empyrion - Galactic Survival"): (2, 2),
    ("Satisfactory", "Farlight Explorers"): (1, 1),
    ("Satisfactory", "A World Reforged"): (1, 1),
    ("Satisfactory", "Caves of Qud"): (1, 1),
    ("Satisfactory", "DwarfCorp"): (1, 1),
    ("Satisfactory", "Chris Sawyer's Locomotion™"): (1, 1),
    ("Satisfactory", "Hexters"): (1, 1),
    ("Satisfactory", "Starship EVO"): (2, 2),
    ("Satisfactory", "Planetbase"): (2, 2),
    ("Satisfactory", "Eco"): (3, 3),
    ("Satisfactory", "Mercury Fallen"): (1, 1),
    ("Satisfactory", "Dwarrows"): (1, 1),
    # Satisfactory
    ("Satisfactory", "Factorio"): (3, 3),

    # New R2 pairs
    ("Don't Starve", "Caves of Qud"): (2, 2),
    ("Don't Starve", "Kynseed"): (2, 2),
    ("Tabletop Simulator", "RimWorld"): (1, 1),
    ("Tabletop Simulator", "Caves of Qud"): (0, 0),
    ("Stardew Valley", "Kynseed"): (3, 3),
}


def main():
    df = pd.read_excel(SRC)
    filled = 0
    for i, row in df.iterrows():
        key = (row["anchor_name"], row["candidate_name"])
        if key in SCORES:
            rel, conf = SCORES[key]
            df.at[i, "relevance"] = rel
            df.at[i, "recommendation_confidence"] = conf
            filled += 1
        else:
            print(f"MISS: {key}")
    df.to_excel(SRC, index=False, engine="openpyxl")
    print(f"filled {filled}/{len(df)} rows -> {SRC}")


if __name__ == "__main__":
    main()

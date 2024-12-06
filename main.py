from ortools.linear_solver import pywraplp
import pandas as pd
from tqdm import tqdm

# Load the input CSV files
players = pd.read_csv('players.csv')  # All players
existing_cards = pd.read_csv('existing_cards.csv')  # Owned cards
rewards = pd.read_csv('rewards.csv')  # Rewards for score thresholds

# Define budget
budget = 0.1  # Budget for buying new players

# Maximum number of decks we want to generate
num_decks = 7

# Create the MIP solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Define rarity multipliers
rarity_multipliers = {
    'Common': 1,
    'Rare': 1.5,
    'Epic': 2,
}

# Precompute player score and price values, including existing cards
def precompute_player_values(players, existing_cards):
    player_values = []
    
    # Add owned cards with zero price, applying the appropriate multiplier based on rarity
    for _, card in existing_cards.iterrows():
        rarity = card['Rarity']
        multiplier = rarity_multipliers[rarity]  # Get the multiplier for the card's rarity
        player_values.append({
            'name': card['Name'],
            'rarity': rarity,
            'lower_score': card['Median Score'] * multiplier * 0.75,  # Apply rarity multiplier to Lower Range Score
            'median_score': card['Median Score'] * multiplier,  # Apply rarity multiplier to Median Score
            'upper_score': card['Upper Range'] * multiplier,  # Apply rarity multiplier to Upper Range Score
            'price': 0,  # Existing cards have zero price
            'is_existing': True  # Flag to indicate this is an existing card
        })
    
    # Add all available players with their respective prices and calculated scores
    for _, player in players.iterrows():
        for rarity, multiplier in rarity_multipliers.items():
            player_values.append({
                'name': player['Name'],
                'rarity': rarity,
                'lower_score': player['Lower Range'] * multiplier,  # Apply rarity multiplier to Lower Range Score
                'median_score': player['Median Score'] * multiplier,  # Apply rarity multiplier to Median Score
                'upper_score': player['Upper Range'] * multiplier,  # Apply rarity multiplier to Upper Range Score
                'price': player['Floor'] * (1 if rarity == 'Common' else 4 if rarity == 'Rare' else 18),
                'is_existing': False  # Flag to indicate this is a new card
            })
    
    return player_values

player_values = precompute_player_values(players, existing_cards)

# Rewards thresholds data
reward_thresholds = []
for _, row in rewards.iterrows():
    reward_thresholds.append({
        'threshold': row['Threshold'],
        'fan_points': row['Fan_Points'],
        'cards': row['Cards'],
        'gold': row['Gold']
    })

# Fix: Get the maximum threshold the score falls into
def get_max_threshold_rewards(score, reward_type):
    # Find the highest threshold the score qualifies for
    applicable_threshold = None
    for reward in reward_thresholds:
        if score >= reward['threshold']:
            applicable_threshold = reward  # Keep updating until we get the highest one
    
    if applicable_threshold:
        return applicable_threshold[reward_type]
    
    return 0  # Return 0 if no threshold is met

# Decision variables: x[i][j] is 1 if player i is in deck j, 0 otherwise
x = []
for i in range(len(player_values)):
    x.append([solver.IntVar(0, 1, f'x[{i}][{j}]') for j in range(num_decks)])

# Decision variables: y[j] is 1 if deck j is selected, 0 otherwise
y = [solver.IntVar(0, 1, f'y[{j}]') for j in range(num_decks)]

# Objective function: maximize fan points, cards, or gold
solver.Maximize(
    solver.Sum(y[j] for j in range(num_decks))  # We'll calculate rewards based on Lower Range Scores later
)

# Constraints: Ensure each deck has exactly 5 players
for j in range(num_decks):
    solver.Add(solver.Sum(x[i][j] for i in range(len(player_values))) == 5)

# Constraints: Ensure total score for each deck meets the threshold using **Lower Range Score** ONLY
for j in range(num_decks):
    solver.Add(
        solver.Sum(player_values[i]['lower_score'] * x[i][j] for i in range(len(player_values))) >= reward_thresholds[0]['threshold'] * y[j]
    )

# Constraints: Ensure total cost for all decks is within the budget
solver.Add(
    solver.Sum(player_values[i]['price'] * x[i][j] for i in range(len(player_values)) for j in range(num_decks)) <= budget
)

# Constraint: Ensure unique player names within a deck
# Map player names to their indices in the player_values array
player_name_to_indices = {}
for i, player in enumerate(player_values):
    if player['name'] not in player_name_to_indices:
        player_name_to_indices[player['name']] = []
    player_name_to_indices[player['name']].append(i)

# For each deck, ensure that each player name appears at most once
for j in range(num_decks):
    for player_name, indices in player_name_to_indices.items():
        solver.Add(solver.Sum(x[i][j] for i in indices) <= 1)  # A player can only appear once per deck

# Constraints: Ensure each existing card can only be used once across all decks
for i, player in enumerate(player_values):
    if player['is_existing']:
        solver.Add(
            solver.Sum(x[i][j] for j in range(num_decks)) <= 1  # An existing card can only be used in one deck
        )

# Solve the problem with progress tracking
with tqdm(total=100, desc="Solving the optimization problem") as pbar:
    status = solver.Solve()
    pbar.update(100)

# Process the results
if status == pywraplp.Solver.OPTIMAL:
    total_fan_points = 0
    total_cards = 0
    total_gold = 0
    total_cost = 0
    new_players_list = []  # List to store new players
    
    print('Solution found!')
    
    for j in range(num_decks):
        if y[j].solution_value() == 1:
            print(f'Deck {j+1}:')
            deck_cost = 0
            deck_lower_score = 0
            deck_median_score = 0
            deck_upper_score = 0
            deck_new_players = []  # List to hold new players for this deck
            
            for i in range(len(player_values)):
                if x[i][j].solution_value() == 1:
                    player = player_values[i]
                    # Mark if player is new
                    player_status = "NEW" if not player["is_existing"] else "EXISTING"
                    # Display Median and Upper Range Scores for Output, but logic still uses Lower Score
                    print(f'  Player: {player["name"]} - {player["rarity"]} (Lower Score: {player["lower_score"]}, Median Score: {player["median_score"]}, Upper Score: {player["upper_score"]}, Price: {player["price"]}, Status: {player_status})')
                    
                    # Collect new players for the list
                    if player_status == "NEW":
                        deck_new_players.append(f"{player['name']} - {player['rarity']} (Price: {player['price']})")
                    
                    deck_cost += player['price']
                    deck_lower_score += player['lower_score']
                    deck_median_score += player['median_score']
                    deck_upper_score += player['upper_score']
            
            total_cost += deck_cost
            # Calculate rewards based on Lower Range Score
            fan_points = get_max_threshold_rewards(deck_lower_score, 'fan_points')
            cards = get_max_threshold_rewards(deck_lower_score, 'cards')
            gold = get_max_threshold_rewards(deck_lower_score, 'gold')
            total_fan_points += fan_points
            total_cards += cards
            total_gold += gold
            print(f'  Deck Cost: {deck_cost}, Score: {deck_lower_score} ({deck_median_score} - {deck_upper_score}), Fan Points: {fan_points}, Cards: {cards}, Gold: {gold}\n')
            
            # Add this deck's new players to the overall list
            if deck_new_players:
                new_players_list.append(f"Deck {j+1}: {', '.join(deck_new_players)}")
    
    # Output all the new players after all decks have been processed
    print("\nNew Players to Purchase:")
    for deck_info in new_players_list:
        print(deck_info)
    
    # Print total rewards and cost at the end
    print(f'\nTotal Fan Points: {total_fan_points}')
    print(f'Total Cards: {total_cards}')
    print(f'Total Gold: {total_gold}')
    print(f'Total Cost: {total_cost}')
else:
    print('No optimal solution found.')


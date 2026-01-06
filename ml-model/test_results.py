#!/usr/bin/env python3
"""
NBA Game Results Diagnostic Script
Tests game_results.json data integrity and calculates correct metrics
"""

import json
import os
import sys
from datetime import datetime

RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'game_results.json')

def load_data():
    """Load game results JSON"""
    if not os.path.exists(RESULTS_PATH):
        print(f"✗ File not found: {RESULTS_PATH}")
        print("\nRun: python update_games.py")
        return None

    try:
        with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def calculate_metrics(recent_results):
    """Calculate correct metrics from recent results"""
    if not recent_results:
        return {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'accuracy': 0.0,
            'games_with_results': 0,
            'games_without_results': 0
        }

    wins = 0
    losses = 0
    games_with_results = 0
    games_without_results = 0

    for game in recent_results:
        result = game.get('result')

        if result == 'win':
            wins += 1
            games_with_results += 1
        elif result == 'loss':
            losses += 1
            games_with_results += 1
        else:
            games_without_results += 1

    total = wins + losses
    accuracy = (wins / total * 100) if total > 0 else 0.0

    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'accuracy': accuracy,
        'games_with_results': games_with_results,
        'games_without_results': games_without_results
    }


def main():
    print("=" * 90)
    print("NBA GAME RESULTS - DIAGNOSTIC TEST")
    print("=" * 90)

    # Load data
    data = load_data()
    if not data:
        return False

    recent_results = data.get('recent_results', [])
    upcoming_games = data.get('upcoming_games', [])
    metadata = data.get('metadata', {})

    print("\n[DATA] Loading Results:")
    print(f"  Recent games: {len(recent_results)}")
    print(f"  Upcoming games: {len(upcoming_games)}")
    print(f"  Last updated: {metadata.get('last_updated', 'N/A')}")

    # Calculate metrics
    metrics = calculate_metrics(recent_results)

    print("\n[METRICS] Calculated from Recent Results:")
    print(f"  Total games: {metrics['total']}")
    print(f"  Wins: {metrics['wins']}")
    print(f"  Losses: {metrics['losses']}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Games with result: {metrics['games_with_results']}")
    print(f"  Games without result: {metrics['games_without_results']}")

    print("\n[METADATA] From JSON (may be inaccurate):")
    print(f"  accuracy_all_time_percent: {metadata.get('accuracy_all_time_percent', 'N/A')}")
    print(f"  accuracy_last_20_percent: {metadata.get('accuracy_last_20_percent', 'N/A')}")
    print(f"  total_games_tracked: {metadata.get('total_games_tracked', 'N/A')}")
    print(f"  correct_predictions: {metadata.get('correct_predictions', 'N/A')}")
    print(f"  incorrect_predictions: {metadata.get('incorrect_predictions', 'N/A')}")
    print(f"  record_wins: {metadata.get('record_wins', 'N/A')}")
    print(f"  record_losses: {metadata.get('record_losses', 'N/A')}")

    # Display recent results
    print("\n[RECENT RESULTS] Last 20 Games:")
    print("-" * 90)
    print(f"{'Date':<12} {'Away':<15} {'Home':<15} {'Score':<10} {'Actual':<12} {'Predicted':<12} {'Result':<8}")
    print("-" * 90)

    for idx, game in enumerate(recent_results[::-1], 1):  # Reverse to show newest first
        date = game.get('date', 'N/A')
        away = game.get('away_team', 'N/A')[:14]
        home = game.get('home_team', 'N/A')[:14]
        score = f"{game.get('away_score', '?')}-{game.get('home_score', '?')}"
        actual = game.get('actual_winner', 'N/A')
        predicted = game.get('predicted', 'N/A')
        result = game.get('result', 'N/A')

        print(f"{date:<12} {away:<15} {home:<15} {score:<10} {actual:<12} {predicted:<12} {result:<8}")

    # Display upcoming games
    print("\n[UPCOMING GAMES] Next Games:")
    print("-" * 90)
    print(f"{'Date':<22} {'Away':<15} {'Home':<15} {'Prediction':<12} {'Confidence':<10}")
    print("-" * 90)

    for idx, game in enumerate(upcoming_games[:7], 1):  # Show next 7
        date = game.get('date', 'N/A')[:22]
        away = game.get('away_team', 'N/A')[:14]
        home = game.get('home_team', 'N/A')[:14]
        predicted = game.get('predicted', 'N/A')
        confidence = f"{game.get('confidence', 0) * 100:.1f}%"

        print(f"{date:<22} {away:<15} {home:<15} {predicted:<12} {confidence:<10}")

    print("\n" + "=" * 90)
    print("SUMMARY:")
    print(f"  ✓ CORRECT Record: {metrics['wins']}-{metrics['losses']} ({metrics['accuracy']:.2f}%)")
    print(f"  ✓ Recent results: {len(recent_results)} games")
    print(f"  ✓ Upcoming games: {len(upcoming_games)} games")
    print("=" * 90)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
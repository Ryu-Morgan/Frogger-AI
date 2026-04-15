# Frogger Reinforcement Learning Environment

A Frogger game engine built for training and testing reinforcement learning agents, specifically Q-learning. You can play it yourself, watch an AI agent play, or train your own agent from scratch.

---

## What Is This?

This project has two parts:

**1. A Frogger Game Engine** — a fully playable Frogger clone that can run with graphics . It handles sprites, terrain, collisions, scoring, etc.

**2. A Q-Learning Agent** — an AI agent that learns to play the game using a Q-table stored as a JSON file. The agent observes the game state, picks actions, and gradually improves its strategy over many episodes.

---

## Project Structure

```

---

## Quick Start

### Prerequisites/Module Installations

```bash
pip install arcade
```

### Command to play as a human

```bash
python main.py player=human screen=medium
```

Use arrow keys to move. Press `Q` or `Escape` to quit.

### Command to watch the AI agent play

```bash
python main.py player=agent screen=medium speed=slow
```

### Command to train the agent (headless)

```bash
python main.py player=agent screen=medium train=q steps=10000 output=text speed=fast
```

---

## Command Line Arguments

---

## How the Game Works

### Screen Layout

The game grid is defined as a string of characters. Each row type has a meaning:

| Character | Terrain | Meaning                         |
| --------- | ------- | ------------------------------- |
| `E`       | Goal    | Top row — reach here to score   |
| `S`       | Safe    | Solid ground, safe to stand     |
| `~`       | Water   | Deadly unless standing on a log |
| `-`       | Road    | Safe unless a car hits you      |
| `K/L/M`   | Log     | Floats left; frog rides it      |
| `T`       | Turtle  | Floats right; frog rides it     |
| `A/B`     | Car     | Moves left — deadly on contact  |
| `C/D`     | Car     | Moves right — deadly on contact |

### Scoring

- Reaching the goal: **50 points + 20 × remaining timer**
- Timer runs for 30 seconds per life
- Death (water, car, timeout): **-10 reward** (during training)

### Example of State Encoding

The game encodes the full screen as a string like:

```
EEEEEEEEEEEEEEEE|~~~KLLLM~~~KLLLM~~|...$goal=110
```

The `$` separates the grid from the note (`goal=<score>`, `done`, or empty).

---

## How the Q-Learning Agent Works

Description: A Q-learning agent is a type of reinforcement learning agent that learns how to act optimally in an environment by trial and error, without needing a model of that environment.:

### The Q-Table

The agent maintains a dictionary mapping _state keys_ → _action values_:

```json
{
  "[[_~_": { "u": 12.3, "d": -5.0, "l": 0.0, "r": 8.1, "_": 0.0 },
  ...
}
```

The key is a compact summary of what the frog sees. The values represent the expected future reward for each action (`u`p, `d`own, `l`eft, `r`ight, `_` stay).

### Current State Key

The default key uses **5 characters**: the 3 cells directly above the frog, plus the cell to its left and right:

```
[above-left] [above] [above-right]
[left]         F      [right]
```

This tells the agent what's coming and what's beside it — enough to avoid immediate danger.

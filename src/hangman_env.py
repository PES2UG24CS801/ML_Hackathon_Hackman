# src/hangman_env.py
import numpy as np
import random
from typing import List, Set, Optional

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
INDEX = {c: i for i, c in enumerate(ALPHABET)}
MAX_LIVES = 6

class HangmanEnv:
    def __init__(self, words: List[str], hmm, max_len: int = None):
        self.words = words
        self.hmm = hmm
        self.max_len = max_len or max(len(w) for w in words)

    def reset(self):
        self.word = random.choice(self.words)
        self.length = len(self.word)
        self.mask = [None] * self.length
        self.guessed = set()
        self.lives = MAX_LIVES
        self.done = False
        return self.get_state()

    def get_state(self) -> np.ndarray:
        state = []
        # 1. Mask (padded)
        for t in range(self.max_len):
            if t >= self.length:
                state.extend([0] * 27)
            elif self.mask[t] is None:
                state.extend([0] * 26 + [1])
            else:
                vec = [0] * 26
                vec[INDEX[self.mask[t]]] = 1
                state.extend(vec + [0])
        # 2. Guessed letters
        state.extend([1 if c in self.guessed else 0 for c in ALPHABET])
        # 3. Lives
        lives_vec = [0] * (MAX_LIVES + 1)
        lives_vec[self.lives] = 1
        state.extend(lives_vec)
        # 4. HMM probabilities
        probs = self.hmm.get_letter_probs(self.mask, self.guessed)
        state.extend(probs)
        return np.array(state, dtype=np.float32)

    def step(self, action: int):
        letter = ALPHABET[action]
        if letter in self.guessed:
            reward = -2
        else:
            self.guessed.add(letter)
            found = False
            for i in range(self.length):
                if self.word[i] == letter:
                    self.mask[i] = letter
                    found = True
            reward = 1 if found else -1
            if not found:
                self.lives -= 1

        won = all(c is not None for c in self.mask)
        lost = self.lives <= 0
        self.done = won or lost
        if won: reward += 10
        if lost: reward -= 10
        return self.get_state(), reward, self.done

    def render(self):
        mask_str = ' '.join(c if c else '_' for c in self.mask)
        print(f"Word: {mask_str} | Lives: {self.lives} | Guessed: {sorted(self.guessed)}")
import sys
from collections import Counter


class Terminal:
    def __init__(self, words):
        len_mode, count = Counter(map(len, words)).most_common(1)[0]
        if count != len(words):
            bad_words = [w for w in words if len(w) != len_mode]
            raise Exception("Some words have different lengths, namely:\n\t{}".format(bad_words))
        self.words = [w.upper() for w in words]

    def letters_common(self, word1, word2):
        return sum(a == b for a, b in zip(word1, word2))

    def attempted(self, word, correct):
        word = word.upper()
        accept_words = []
        for w in self.words:
            if self.letters_common(w, word) == correct:
                accept_words.append(w)
        self.words = accept_words
        if accept_words == []:
            return None
        else:
            return self.words[-1]

    def automate(self):
        guess = self.words.pop()
        while True:
            print("Next guess:{:>10}".format(guess))
            while True:
                try:
                    correct = int(input("How many were correct? "))
                    break
                except KeyboardInterrupt:
                    sys.exit()
                except ValueError:
                    print("Please input an integer...")
            guess = self.attempted(guess, correct)
            if guess is None:
                print("There are no guesses left")
                return
            else:
                self.words.pop()


if __name__ == "__main__":
    # wl = ['doctrines', 'batteries', 'scorpions', 'generated', 'occasions',
    #     'occupants', 'officials', 'intricate', 'deformity', 'brutality',
    #     'carriages', 'accompany', 'interface']
    # t = Terminal(wl)
    # t.automate()
    # # print(t.letters_common('deadens', 'petrols'))
    pass

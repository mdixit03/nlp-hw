# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """


        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?
        # for thing in question: 
        #     print(thing)
        # print(question['subcategory'])
        # print(f"Question type: {type(question)}")
        # print(f"Run type: {type(run)}")
        # print(f"Guess type: {type(guess)}")
        # print("Question: " + question)
        # print("Run: " + run)
        # print("Guess: " + guess)
        question_length = len(question['tokenizations'])
        guess_length = len(guess) 

        # How many words long is the question?


        # How many characters long is the guess?
        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:   
            yield ("word", len(run.split())) #length of run in words
            yield ("char", len(run))  #length of run in chars
            yield ("question", question_length) #length of question in tokens
            #yield ("guess", guess_length) #length of guess in chars

            
class FrequencyFeature(Feature):
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer
        
    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
        for ii in questions:
            self.counts[self.normalize(ii["page"])] += 1
            
    def __call__(self, question, run, guess, guess_history, guesses):
        # We only use question, run, and guess (same as before)
        # guess_history and guesses are ignored since we don't need them
        
        frequency_value = log(1 + self.counts[self.normalize(guess)])
        yield ("guess_frequency", frequency_value) # <--- Changed this yield instead of return

class DisambiguatorFeature(Feature):
    """
    Is there a parentheses disambiguator?
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ("disambiguator", ("(" in guess and ")" in guess))
        
class CategoryFeature(Feature):
    """
    Category of the question
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ("Category", question['category'])

class SubcategoryFeature(Feature):
    """
    Category of the question
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ("Subcategory", question['subcategory'])

class YearFeature(Feature):
    """
    Category of the question
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ("year", question['year'])


class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('true', len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))


if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from params import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    add_guesser_params(parser)
    add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags)
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)

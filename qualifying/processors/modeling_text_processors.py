import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class TextProcessor(ABC):
    """
    A metaclass used for type checking.
    """

    @abstractmethod
    def __call__(self, text: str):
        raise NotImplementedError


@dataclass
class DuplicatedWhitespaceRemovingTextProcessor(TextProcessor):
    """
    Removes duplicated whitespaces in any given string.
    """
    whitespace_pattern: re.Pattern = field(init=False)

    def __post_init__(self):
        self.whitespace_pattern = re.compile("\\s+")

    def __call__(self, text: str):
        processed = self.whitespace_pattern.sub(" ", text.strip())

        return processed


@dataclass
class ChoiceSelectionTextProcessor(TextProcessor):
    """
    Select one between two choices wrapped and distinguished by parentheses and slash.
    If former choice violates the condition, the latter is selected whether it satisfies the condition or not.
    Otherwise, the former is selected.

    ex) 'Choose (one)/(#two) between choices.' -> 'Choose one between choices.' if condition: '[^ a-z]+'.
    """
    condition: str = field(
        metadata={"help": "A regex format condition. The choice which violates this condition is dropped."}
    )

    choice_pattern: re.Pattern = field(init=False)
    exclusive_pattern: re.Pattern = field(init=False)

    def __post_init__(self):
        self.choice_pattern = re.compile("\\([^()]+\\)/\\([^()]+\\)")
        self.exclusive_pattern = re.compile(self.condition)

    def __call__(self, text: str):
        def choose(former: str, latter: str):
            if self.exclusive_pattern.findall(former):
                return latter
            return former

        choices = [
            choose(*map(lambda c: c[1:-1], s.split("/")))
            for s in self.choice_pattern.findall(text)
        ]
        blanked_sentences = self.choice_pattern.split(text)

        processed = ""
        for blanked_sentence, choice in zip(blanked_sentences, choices):
            processed += blanked_sentence + choice
        processed += blanked_sentences[-1]

        return processed


@dataclass
class AnonymityMaskTextProcessor(TextProcessor):
    def __call__(self, text: str):
        pass


class SequentialTextProcessor(TextProcessor):
    """
    Sequentially executes passed text processors when it's called.
    All passed text processors must be a class inherited from TextProcessor.
    """
    def __init__(self, *args: TextProcessor):
        for processor in args:
            if not isinstance(processor, TextProcessor):
                raise ValueError(
                    f"Any argument for {self.__class__} must be a class inherited from TextProcessor."
                    f" Found {type(processor)} instead."
                )

        self.processors = args

    def __call__(self, text: str):
        for processor in self.processors:
            text = processor(text)

        return text

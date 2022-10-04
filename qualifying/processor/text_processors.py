import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


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
        metadata={"help": "A condition in regex format. The choice which violates this condition is dropped."}
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
class SubstituteExceptTextProcessor(TextProcessor):
    condition: str = field(
        metadata={
            "help": "A condition in regex format. When text processor is called,"
                    " it substitutes letters according to this condition."
        }
    )
    substitute_to: str = field(default="", metadata={"help": "String to substitute if the condition is valid."})
    exceptions: Optional[List[str]] = field(default=None, metadata={"help": "Identifies exceptions."})

    exception_pattern: re.Pattern = field(init=False)
    substitution_pattern: re.Pattern = field(init=False)

    def __post_init__(self):
        self.exception_pattern = re.compile("|".join(self.exceptions))
        self.substitution_pattern = re.compile(self.condition)

    def __call__(self, text: str):
        exceptions = self.exception_pattern.findall(text)
        blanked_sentences = [
            self.substitution_pattern.sub(self.substitute_to, s)
            for s in self.exception_pattern.split(text)
        ]

        processed = ""
        for blanked_sentence, exception in zip(blanked_sentences, exceptions):
            processed += blanked_sentence + exception
        processed += blanked_sentences[-1]

        return processed


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

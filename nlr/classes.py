from dataclasses import dataclass, field


@dataclass(order=True, unsafe_hash=True)
class Atom:
    order: int = field(init=True, compare=True, hash=True)
    var: str = field(default='x', compare=True, hash=True)
    noun: str = field(default=None, compare=False, hash=False)

    @property
    def name(self):
        return f"{self.var}{self.order}"

    def __str__(self):
        return self.name

    def __int__(self):
        return self.order


@dataclass(eq=True, unsafe_hash=True)
class Literal:
    polarity: bool = field(compare=True, hash=True)
    atom: Atom = field(compare=True, hash=True)

    def __str__(self):
        return f"{'+' if self.polarity else '-'}{str(self.atom)}"
